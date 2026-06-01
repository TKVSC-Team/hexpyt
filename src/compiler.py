"""
compiler.py — A proper recursive-descent Hexpat → Python compiler.
Replaces the old hexpyt line-by-line transpiler.

Supports:
  - struct / enum / bitfield declarations
  - namespace unwrapping
  - generic templates (struct S<T>)
  - struct inheritance (struct B : A)
  - if / else / while inside struct bodies
  - `T x @ expr;` (offset placement)
  - `T x[N];` (arrays) and `T x[];` (null-terminated strings)
  - padding[N]
  - using aliases
  - [[attribute]] stripping
  - fn function declarations (skipped, not executable in our runtime)
  - #pragma / #include / #define (skipped)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# TOKENS
# ─────────────────────────────────────────────────────────────────────────────

TK_IDENT = 'IDENT'
TK_INT   = 'INT'
TK_FLOAT = 'FLOAT'
TK_STR   = 'STR'
TK_OP    = 'OP'    # multi-char operators (::, &&, ||, ==, !=, <=, >=, <<, >>)
TK_SYM   = 'SYM'  # single-char symbols
TK_HASH  = 'HASH' # preprocessor line (skipped)
TK_EOF   = 'EOF'

KEYWORDS = frozenset({
    'struct', 'enum', 'bitfield', 'namespace', 'fn', 'using',
    'if', 'else', 'while', 'break', 'continue', 'return',
    'auto', 'be', 'le', 'padding', 'in', 'type',
})

# Primitives whose Python class names differ from their hexpat names
PRIM_MAP: Dict[str, str] = {
    'float':   'Float',
    'bool':    'Bool',
    'bool8':   'Bool',
    'bool16':  'Bool',
    'bool32':  'Bool',
    'double':  'double',
    'char':    'char',
    'char16':  'char16',
}

MULTI_CHAR_OPS = {
    '::', '&&', '||', '!=', '==', '<=', '>=', '<<', '>>',
    '->', '+=', '-=', '*=', '/=', '|=', '&=', '^=',
}

@dataclass
class Token:
    type: str
    value: str
    line: int

    def __repr__(self) -> str:
        return f'Token({self.type}, {self.value!r}, L{self.line})'


# ─────────────────────────────────────────────────────────────────────────────
# LEXER
# ─────────────────────────────────────────────────────────────────────────────

class LexError(Exception):
    pass


class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.pos  = 0
        self.line = 1

    def _peek(self, off: int = 0) -> str:
        p = self.pos + off
        return self.src[p] if p < len(self.src) else '\0'

    def _advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
        return ch

    def _skip_ws_and_comments(self) -> None:
        while self.pos < len(self.src):
            ch = self._peek()
            if ch in ' \t\r\n':
                self._advance()
            elif ch == '/' and self._peek(1) == '/':
                while self.pos < len(self.src) and self._peek() != '\n':
                    self._advance()
            elif ch == '/' and self._peek(1) == '*':
                self._advance(); self._advance()
                while self.pos < len(self.src):
                    if self._peek() == '*' and self._peek(1) == '/':
                        self._advance(); self._advance()
                        break
                    self._advance()
            else:
                break

    def _read_string(self, delim: str) -> str:
        s = ''
        while self.pos < len(self.src):
            ch = self._advance()
            if ch == '\\':
                esc = self._advance()
                s += {'n': '\n', 't': '\t', 'r': '\r'}.get(esc, esc)
            elif ch == delim:
                return s
            else:
                s += ch
        raise LexError(f'Line {self.line}: Unterminated string')

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            self._skip_ws_and_comments()
            if self.pos >= len(self.src):
                tokens.append(Token(TK_EOF, '', self.line))
                break
            line = self.line
            ch   = self._peek()

            # Preprocessor directive → skip
            if ch == '#':
                self._advance()
                self._skip_ws_and_comments()
                directive = ''
                while self.pos < len(self.src) and self._peek() not in '\r\n':
                    if self._peek() == '\\' and self._peek(1) == '\n':
                        self._advance(); self._advance()
                        continue
                    directive += self._advance()
                tokens.append(Token(TK_HASH, directive.strip(), line))
                continue

            # [[attributes]] → skip entirely
            if ch == '[' and self._peek(1) == '[':
                self._advance(); self._advance()
                depth = 1
                while self.pos < len(self.src):
                    c = self._peek()
                    if c == '[' and self._peek(1) == '[':
                        depth += 1; self._advance(); self._advance()
                    elif c == ']' and self._peek(1) == ']':
                        depth -= 1; self._advance(); self._advance()
                        if depth == 0:
                            break
                    else:
                        self._advance()
                continue

            # String literals
            if ch in ('"', "'"):
                self._advance()
                s = self._read_string(ch)
                tokens.append(Token(TK_STR, s, line))
                continue

            # Numbers
            if ch.isdigit():
                num = ''
                is_float = False
                if ch == '0' and self._peek(1) in 'xX':
                    num += self._advance() + self._advance()
                    while self._peek().isdigit() or self._peek() in 'abcdefABCDEF_':
                        c = self._advance()
                        if c != '_':
                            num += c
                elif ch == '0' and self._peek(1) in 'bB':
                    num += self._advance() + self._advance()
                    while self._peek() in '01_':
                        c = self._advance()
                        if c != '_':
                            num += c
                else:
                    while self._peek().isdigit():
                        num += self._advance()
                    if self._peek() == '.':
                        is_float = True
                        num += self._advance()
                        while self._peek().isdigit():
                            num += self._advance()
                tokens.append(Token(TK_FLOAT if is_float else TK_INT, num, line))
                continue

            # Identifiers / keywords
            if ch.isalpha() or ch == '_':
                ident = ''
                while self._peek().isalnum() or self._peek() == '_':
                    ident += self._advance()
                tokens.append(Token(TK_IDENT, ident, line))
                continue

            # Multi-char operators
            two = ch + self._peek(1)
            if two in MULTI_CHAR_OPS:
                self._advance(); self._advance()
                tokens.append(Token(TK_OP, two, line))
                continue

            # Single-char symbol
            self._advance()
            tokens.append(Token(TK_SYM, ch, line))

        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# AST
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TypeRef:
    name: str
    template_args: List['TypeRef'] = dc_field(default_factory=list)
    is_auto: bool = False  # True if this was an `auto` template arg

    def py_name(self) -> str:
        """Return the Python class name for this type."""
        n = PRIM_MAP.get(self.name, self.name)
        if self.template_args:
            args = '_'.join(a.py_name() for a in self.template_args)
            return f'{n}_{args}'
        return n


@dataclass
class SimpleField:
    type_ref: TypeRef
    name: str
    array_expr:   Optional[str] = None   # None=scalar, ''=dynamic (T x[]), str=count expr
    offset_expr:  Optional[str] = None   # None=sequential, str=absolute offset expr
    is_auto_array: bool = False          # True for `T x[]`
    value_expr:   Optional[str] = None   # `= expr` initializer; if '$', capture position only


@dataclass
class PaddingField:
    size_expr: str
    index: int


@dataclass
class IfBlock:
    cond_expr: str
    then_body: List
    else_body: List


@dataclass
class WhileBlock:
    cond_expr: str
    body: List


@dataclass
class BreakStmt:
    pass


@dataclass
class StructDecl:
    name: str
    template_params: List[str]
    is_auto_template: bool
    base_type: Optional[str]
    body: List
    line: int = 0


@dataclass
class EnumDecl:
    name: str
    base_type: str
    values: List[Tuple[str, Optional[str]]]
    line: int = 0


@dataclass
class BitfieldDecl:
    name: str
    fields: List[Tuple[str, str]]
    line: int = 0


@dataclass
class TopLevelPlacement:
    type_ref: TypeRef
    name: str
    offset_expr: Optional[str]
    line: int = 0


@dataclass
class UsingDecl:
    alias: str
    target: Optional[str]


@dataclass
class NamespaceDecl:
    name: str
    body: List


@dataclass
class FunctionDecl:
    name: str


@dataclass
class Program:
    decls: List


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos    = 0

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, off: int = 1) -> Token:
        p = self.pos + off
        return self.tokens[p] if p < len(self.tokens) else self.tokens[-1]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def _expect(self, type_: str, value: str = None) -> Token:
        t = self._cur()
        if t.type != type_:
            raise ParseError(
                f'Line {t.line}: Expected {type_} but got {t.type!r} ({t.value!r})')
        if value is not None and t.value != value:
            raise ParseError(
                f'Line {t.line}: Expected {value!r} but got {t.value!r}')
        return self._advance()

    def _eat(self, type_: str, value: str = None) -> bool:
        t = self._cur()
        if t.type == type_ and (value is None or t.value == value):
            self._advance()
            return True
        return False

    # ── Expression ──────────────────────────────────────────────────────────

    def _parse_expr(self, stop_syms: set, stop_depth: int = 0) -> str:
        """Collect tokens as a raw expression string until a stop symbol at brace depth 0."""
        parts = []
        depth = 0
        while True:
            t = self._cur()
            if t.type == TK_EOF:
                break
            if t.type == TK_SYM:
                if t.value in ('{', '(', '['):
                    depth += 1
                elif t.value in ('}', ')', ']'):
                    if depth == stop_depth and t.value in stop_syms:
                        break
                    depth = max(0, depth - 1)
                elif t.value in stop_syms and depth == stop_depth:
                    break
                parts.append(t.value)
            elif t.type == TK_OP:
                op_map = {'&&': 'and', '||': 'or', '::': '.'}
                parts.append(op_map.get(t.value, t.value))
            elif t.type == TK_IDENT:
                parts.append(t.value)
            elif t.type in (TK_INT, TK_FLOAT):
                parts.append(t.value)
            elif t.type == TK_STR:
                parts.append(repr(t.value))
            elif t.type == TK_HASH:
                self._advance()
                continue
            else:
                break
            self._advance()
        return ' '.join(parts)

    # ── Type reference ───────────────────────────────────────────────────────

    def _parse_type_ref(self) -> TypeRef:
        """Parse a (possibly namespace-qualified, possibly templated) type reference."""
        name = self._expect(TK_IDENT).value

        # Consume namespace qualifiers: agl::ImageFormat → ImageFormat
        while self._cur().type == TK_OP and self._cur().value == '::':
            self._advance()
            name = self._expect(TK_IDENT).value

        # Template arguments: RelPtr<String>, Array<u32>
        template_args: List[TypeRef] = []
        if self._cur().type == TK_SYM and self._cur().value == '<':
            saved = self.pos
            try:
                self._advance()  # consume <
                while True:
                    if self._cur().type == TK_SYM and self._cur().value == '>':
                        self._advance()
                        break
                    is_auto = False
                    if self._cur().type == TK_IDENT and self._cur().value == 'auto':
                        self._advance()
                        is_auto = True
                    arg_tr = self._parse_type_ref()
                    arg_tr.is_auto = is_auto
                    template_args.append(arg_tr)
                    if not self._eat(TK_SYM, ','):
                        if self._cur().type == TK_SYM and self._cur().value == '>':
                            self._advance()
                        break
            except ParseError:
                # Backtrack — `<` was probably a comparison operator
                self.pos = saved
                template_args = []

        return TypeRef(name=name, template_args=template_args)

    # ── Struct body ──────────────────────────────────────────────────────────

    def _parse_struct_body(self, padding_counter: List[int]) -> List:
        body = []
        while True:
            t = self._cur()
            if t.type == TK_EOF or (t.type == TK_SYM and t.value == '}'):
                break
            if self._eat(TK_SYM, ';'):
                continue
            if t.type == TK_HASH:
                self._advance()
                continue

            kw = t.value if t.type == TK_IDENT else None

            if kw == 'struct':
                body.append(self._parse_struct_decl())
                continue
            if kw == 'if':
                body.append(self._parse_if_block(padding_counter))
                continue
            if kw == 'while':
                body.append(self._parse_while_block(padding_counter))
                continue
            if kw == 'break':
                self._advance(); self._eat(TK_SYM, ';')
                body.append(BreakStmt())
                continue
            if kw == 'return':
                self._advance()
                self._parse_expr({';'})  # discard
                self._eat(TK_SYM, ';')
                continue
            if kw == 'padding':
                self._advance()
                self._expect(TK_SYM, '[')
                size_expr = self._parse_expr({']'})
                self._expect(TK_SYM, ']')
                self._eat(TK_SYM, ';')
                body.append(PaddingField(size_expr=size_expr.strip(), index=padding_counter[0]))
                padding_counter[0] += 1
                continue

            # Attempt to parse as a field declaration: TypeRef name [...]? [@ expr]? ;
            if t.type == TK_IDENT:
                saved = self.pos
                try:
                    type_ref = self._parse_type_ref()
                    if (self._cur().type == TK_IDENT
                            and self._cur().value not in KEYWORDS):
                        field_name = self._advance().value

                        # Optional default-value initializer: TypeName field = expr;
                        # `= $` means capture current position (no read); other = override after read
                        value_expr = None
                        if self._cur().type == TK_SYM and self._cur().value == '=':
                            self._advance()
                            value_expr = self._parse_expr({';', ','}).strip()

                        # Optional array spec
                        array_expr   = None
                        is_auto_array = False
                        if self._cur().type == TK_SYM and self._cur().value == '[':
                            self._advance()
                            if self._cur().type == TK_SYM and self._cur().value == ']':
                                self._advance()
                                is_auto_array = True
                                array_expr = ''
                            else:
                                array_expr = self._parse_expr({']'}).strip()
                                self._expect(TK_SYM, ']')

                        # Optional @ offset
                        offset_expr = None
                        if self._cur().type == TK_SYM and self._cur().value == '@':
                            self._advance()
                            offset_expr = self._parse_expr({';', ','}).strip()

                        # Emit this field
                        body.append(SimpleField(
                            type_ref=type_ref,
                            name=field_name,
                            array_expr=array_expr,
                            offset_expr=offset_expr,
                            is_auto_array=is_auto_array,
                            value_expr=value_expr,
                        ))

                        # Handle comma-separated multi-declarator: float x, y, z;
                        # Each extra name shares the same type_ref but is a fresh
                        # sequential field (no array/offset/value carry-over).
                        while self._cur().type == TK_SYM and self._cur().value == ',':
                            self._advance()  # consume ','
                            if (self._cur().type == TK_IDENT
                                    and self._cur().value not in KEYWORDS):
                                extra_name = self._advance().value
                                body.append(SimpleField(
                                    type_ref=type_ref,
                                    name=extra_name,
                                    array_expr=None,
                                    offset_expr=None,
                                    is_auto_array=False,
                                    value_expr=None,
                                ))
                            else:
                                break

                        self._eat(TK_SYM, ';')
                        continue
                    else:
                        self.pos = saved
                except ParseError:
                    self.pos = saved

            # Unrecognised statement — skip to next `;` or `}`
            self._skip_stmt()

        return body

    def _skip_stmt(self) -> None:
        depth = 0
        while self._cur().type != TK_EOF:
            t = self._cur()
            if t.type == TK_SYM:
                if t.value == '{':
                    depth += 1
                elif t.value == '}':
                    if depth == 0:
                        return
                    depth -= 1
                elif t.value == ';' and depth == 0:
                    self._advance()
                    return
            self._advance()

    def _parse_if_block(self, padding_counter: List[int]) -> IfBlock:
        self._expect(TK_IDENT, 'if')
        self._expect(TK_SYM, '(')
        cond = self._parse_expr({')'}).strip()
        self._expect(TK_SYM, ')')
        self._expect(TK_SYM, '{')
        then_body = self._parse_struct_body(padding_counter)
        self._expect(TK_SYM, '}')

        else_body: List = []
        if self._cur().type == TK_IDENT and self._cur().value == 'else':
            self._advance()
            if self._cur().type == TK_IDENT and self._cur().value == 'if':
                else_body = [self._parse_if_block(padding_counter)]
            else:
                self._expect(TK_SYM, '{')
                else_body = self._parse_struct_body(padding_counter)
                self._expect(TK_SYM, '}')

        return IfBlock(cond_expr=cond, then_body=then_body, else_body=else_body)

    def _parse_while_block(self, padding_counter: List[int]) -> WhileBlock:
        self._expect(TK_IDENT, 'while')
        self._expect(TK_SYM, '(')
        cond = self._parse_expr({')'}).strip()
        self._expect(TK_SYM, ')')
        self._expect(TK_SYM, '{')
        body = self._parse_struct_body(padding_counter)
        self._expect(TK_SYM, '}')
        return WhileBlock(cond_expr=cond, body=body)

    # ── Top-level declarations ───────────────────────────────────────────────

    def _parse_struct_decl(self) -> StructDecl:
        line = self._cur().line
        self._expect(TK_IDENT, 'struct')
        name = self._expect(TK_IDENT).value

        template_params: List[str] = []
        is_auto_template = False
        if self._cur().type == TK_SYM and self._cur().value == '<':
            self._advance()
            while not (self._cur().type == TK_SYM and self._cur().value == '>'):
                if self._cur().type == TK_EOF:
                    break
                if self._cur().type == TK_IDENT and self._cur().value == 'auto':
                    self._advance()
                    if self._cur().type == TK_IDENT:
                        template_params.append(self._advance().value)
                    is_auto_template = True
                elif self._cur().type == TK_IDENT:
                    template_params.append(self._advance().value)
                self._eat(TK_SYM, ',')
            self._eat(TK_SYM, '>')

        base_type: Optional[str] = None
        if self._cur().type == TK_SYM and self._cur().value == ':':
            self._advance()
            try:
                base_type = self._parse_type_ref().py_name()
            except ParseError:
                pass

        self._expect(TK_SYM, '{')
        body = self._parse_struct_body([0])
        self._expect(TK_SYM, '}')
        self._eat(TK_SYM, ';')

        return StructDecl(
            name=name,
            template_params=template_params,
            is_auto_template=is_auto_template,
            base_type=base_type,
            body=body,
            line=line,
        )

    def _parse_enum_decl(self) -> EnumDecl:
        line = self._cur().line
        self._expect(TK_IDENT, 'enum')
        name = self._expect(TK_IDENT).value
        self._expect(TK_SYM, ':')
        base_type = self._parse_type_ref().py_name()
        self._expect(TK_SYM, '{')

        values: List[Tuple[str, Optional[str]]] = []
        while True:
            t = self._cur()
            if t.type == TK_EOF or (t.type == TK_SYM and t.value == '}'):
                break
            if self._eat(TK_SYM, ','):
                continue
            if t.type == TK_IDENT:
                vname = self._advance().value
                vexpr: Optional[str] = None
                if self._cur().type == TK_SYM and self._cur().value == '=':
                    self._advance()
                    vexpr = self._parse_expr({',', '}'}).strip()
                values.append((vname, vexpr))
            else:
                self._advance()

        self._eat(TK_SYM, '}')
        self._eat(TK_SYM, ';')
        return EnumDecl(name=name, base_type=base_type, values=values, line=line)

    def _parse_bitfield_decl(self) -> BitfieldDecl:
        line = self._cur().line
        self._expect(TK_IDENT, 'bitfield')
        name = self._expect(TK_IDENT).value
        self._expect(TK_SYM, '{')
        fields: List[Tuple[str, str]] = []
        while True:
            t = self._cur()
            if t.type == TK_EOF or (t.type == TK_SYM and t.value == '}'):
                break
            if t.type == TK_IDENT:
                fname = self._advance().value
                self._eat(TK_SYM, ':')
                bits_expr = self._parse_expr({';', ','}).strip()
                self._eat(TK_SYM, ';')
                fields.append((fname, bits_expr))
            else:
                self._advance()
        self._eat(TK_SYM, '}')
        self._eat(TK_SYM, ';')
        return BitfieldDecl(name=name, fields=fields, line=line)

    def _parse_fn_decl(self) -> FunctionDecl:
        self._expect(TK_IDENT, 'fn')
        name = self._expect(TK_IDENT).value
        # Skip params
        self._expect(TK_SYM, '(')
        depth = 1
        while depth > 0 and self._cur().type != TK_EOF:
            v = self._cur().value
            if self._cur().type == TK_SYM and v == '(':
                depth += 1
            elif self._cur().type == TK_SYM and v == ')':
                depth -= 1
            self._advance()
        # Skip body
        if self._cur().type == TK_SYM and self._cur().value == '{':
            self._advance()
            depth = 1
            while depth > 0 and self._cur().type != TK_EOF:
                v = self._cur().value
                if self._cur().type == TK_SYM and v == '{':
                    depth += 1
                elif self._cur().type == TK_SYM and v == '}':
                    depth -= 1
                self._advance()
        self._eat(TK_SYM, ';')
        return FunctionDecl(name=name)

    def _parse_using_decl(self) -> UsingDecl:
        self._expect(TK_IDENT, 'using')
        alias = self._expect(TK_IDENT).value
        target: Optional[str] = None
        if self._cur().type == TK_SYM and self._cur().value == '=':
            self._advance()
            try:
                target = self._parse_type_ref().py_name()
            except ParseError:
                pass
        self._eat(TK_SYM, ';')
        return UsingDecl(alias=alias, target=target)

    def _parse_namespace_decl(self) -> NamespaceDecl:
        self._expect(TK_IDENT, 'namespace')
        # Support multi-component: namespace vfx2::detail
        name = self._expect(TK_IDENT).value
        while self._cur().type == TK_OP and self._cur().value == '::':
            self._advance()
            name += '::' + self._expect(TK_IDENT).value
        self._expect(TK_SYM, '{')
        body = self._parse_program_body()
        self._expect(TK_SYM, '}')
        return NamespaceDecl(name=name, body=body)

    def _parse_program_body(self) -> List:
        decls: List = []
        while True:
            t = self._cur()
            if t.type == TK_EOF:
                break
            if t.type == TK_SYM and t.value == '}':
                break
            if self._eat(TK_SYM, ';'):
                continue
            if t.type == TK_HASH:
                self._advance()
                continue

            if t.type == TK_IDENT:
                kw = t.value
                if kw == 'struct':
                    decls.append(self._parse_struct_decl())
                elif kw == 'enum':
                    decls.append(self._parse_enum_decl())
                elif kw == 'bitfield':
                    decls.append(self._parse_bitfield_decl())
                elif kw == 'namespace':
                    decls.append(self._parse_namespace_decl())
                elif kw == 'fn':
                    decls.append(self._parse_fn_decl())
                elif kw == 'using':
                    decls.append(self._parse_using_decl())
                else:
                    # Top-level placement: TypeRef name [@ expr] ;
                    saved = self.pos
                    try:
                        type_ref = self._parse_type_ref()
                        if (self._cur().type == TK_IDENT
                                and self._cur().value not in KEYWORDS):
                            pname = self._advance().value
                            offset_expr: Optional[str] = None
                            if self._cur().type == TK_SYM and self._cur().value == '@':
                                self._advance()
                                offset_expr = self._parse_expr({';'}).strip()
                            self._eat(TK_SYM, ';')
                            decls.append(TopLevelPlacement(
                                type_ref=type_ref,
                                name=pname,
                                offset_expr=offset_expr,
                                line=t.line,
                            ))
                        else:
                            self.pos = saved
                            self._skip_stmt()
                    except ParseError:
                        self.pos = saved
                        self._skip_stmt()
            else:
                self._advance()
        return decls

    def parse(self) -> Program:
        return Program(decls=self._parse_program_body())


# ─────────────────────────────────────────────────────────────────────────────
# NAMESPACE FLATTENING
# ─────────────────────────────────────────────────────────────────────────────

def flatten_namespaces(decls: List) -> List:
    result: List = []
    for d in decls:
        if isinstance(d, NamespaceDecl):
            result.extend(flatten_namespaces(d.body))
        else:
            result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _collect_type_usages(nodes: List) -> List[TypeRef]:
    refs: List[TypeRef] = []
    for node in nodes:
        if isinstance(node, SimpleField):
            refs.append(node.type_ref)
            refs.extend(_collect_type_usages_in_tr(node.type_ref))
        elif isinstance(node, IfBlock):
            refs.extend(_collect_type_usages(node.then_body))
            refs.extend(_collect_type_usages(node.else_body))
        elif isinstance(node, WhileBlock):
            refs.extend(_collect_type_usages(node.body))
        elif isinstance(node, StructDecl):
            refs.extend(_collect_type_usages(node.body))
        elif isinstance(node, TopLevelPlacement):
            refs.append(node.type_ref)
    return refs


def _collect_type_usages_in_tr(tr: TypeRef) -> List[TypeRef]:
    out = list(tr.template_args)
    for a in tr.template_args:
        out.extend(_collect_type_usages_in_tr(a))
    return out


def _subst_type(tr: TypeRef, param_map: Dict[str, str]) -> TypeRef:
    new_name = param_map.get(tr.name, tr.name)
    new_args = [_subst_type(a, param_map) for a in tr.template_args]
    return TypeRef(name=new_name, template_args=new_args)


def _subst_body(body: List, param_map: Dict[str, str]) -> List:
    result: List = []
    for node in body:
        if isinstance(node, SimpleField):
            result.append(SimpleField(
                type_ref=_subst_type(node.type_ref, param_map),
                name=node.name,
                array_expr=node.array_expr,
                offset_expr=node.offset_expr,
                is_auto_array=node.is_auto_array,
            ))
        elif isinstance(node, IfBlock):
            result.append(IfBlock(
                cond_expr=node.cond_expr,
                then_body=_subst_body(node.then_body, param_map),
                else_body=_subst_body(node.else_body, param_map),
            ))
        elif isinstance(node, WhileBlock):
            result.append(WhileBlock(
                cond_expr=node.cond_expr,
                body=_subst_body(node.body, param_map),
            ))
        elif isinstance(node, StructDecl):
            result.append(StructDecl(
                name=node.name,
                template_params=node.template_params,
                is_auto_template=node.is_auto_template,
                base_type=node.base_type,
                body=_subst_body(node.body, param_map),
                line=node.line,
            ))
        else:
            result.append(node)
    return result


def _flatten_tr(tr: TypeRef) -> TypeRef:
    """Collapse a templated TypeRef to its concrete flat name (no template_args)."""
    if not tr.template_args:
        return tr
    concrete = tr.py_name()
    return TypeRef(name=concrete, template_args=[])


def _flatten_refs_in_body(body: List) -> List:
    result: List = []
    for node in body:
        if isinstance(node, SimpleField):
            result.append(SimpleField(
                type_ref=_flatten_tr(node.type_ref),
                name=node.name,
                array_expr=node.array_expr,
                offset_expr=node.offset_expr,
                is_auto_array=node.is_auto_array,
            ))
        elif isinstance(node, IfBlock):
            result.append(IfBlock(
                cond_expr=node.cond_expr,
                then_body=_flatten_refs_in_body(node.then_body),
                else_body=_flatten_refs_in_body(node.else_body),
            ))
        elif isinstance(node, WhileBlock):
            result.append(WhileBlock(
                cond_expr=node.cond_expr,
                body=_flatten_refs_in_body(node.body),
            ))
        elif isinstance(node, StructDecl):
            result.append(StructDecl(
                name=node.name,
                template_params=node.template_params,
                is_auto_template=node.is_auto_template,
                base_type=node.base_type,
                body=_flatten_refs_in_body(node.body),
                line=node.line,
            ))
        elif isinstance(node, TopLevelPlacement):
            result.append(TopLevelPlacement(
                type_ref=_flatten_tr(node.type_ref),
                name=node.name,
                offset_expr=node.offset_expr,
                line=node.line,
            ))
        else:
            result.append(node)
    return result


def resolve_templates(decls: List) -> List:
    """
    1. Collect all generic struct definitions.
    2. Find all instantiation sites.
    3. Generate concrete class declarations.
    4. Replace all templated TypeRefs with their flat names.
    """
    # Step 1 — collect templates (skip `auto` templates, they can't be resolved statically)
    templates: Dict[str, StructDecl] = {}
    for d in decls:
        if (isinstance(d, StructDecl)
                and d.template_params
                and not d.is_auto_template):
            templates[d.name] = d

    if not templates:
        return _flatten_refs_in_body(decls)

    # Step 2 — find instantiation sites
    all_type_refs = _collect_type_usages(decls)
    # Also scan TopLevelPlacement nodes at top level
    for d in decls:
        if isinstance(d, TopLevelPlacement):
            all_type_refs.append(d.type_ref)

    instantiation_set: set = set()
    for tr in all_type_refs:
        if tr.name in templates and tr.template_args:
            args_key = tuple(a.py_name() for a in tr.template_args)
            instantiation_set.add((tr.name, args_key))

    # Step 3 — generate concrete classes
    generated: Dict[str, StructDecl] = {}
    for (t_name, args_key) in instantiation_set:
        tdef = templates[t_name]
        concrete_name = f"{t_name}_{'_'.join(args_key)}"
        if concrete_name in generated:
            continue
        param_map = {tdef.template_params[i]: args_key[i]
                     for i in range(min(len(tdef.template_params), len(args_key)))}
        concrete_body = _subst_body(tdef.body, param_map)
        generated[concrete_name] = StructDecl(
            name=concrete_name,
            template_params=[],
            is_auto_template=False,
            base_type=tdef.base_type,
            body=concrete_body,
            line=tdef.line,
        )

    # Step 4 — rebuild decl list
    new_decls: List = []
    for d in decls:
        if isinstance(d, StructDecl) and d.name in templates:
            # Emit all concrete instantiations of this template here
            for (t_name, _) in instantiation_set:
                if t_name == d.name:
                    concrete_name = f"{t_name}_{'_'.join(_)}"
                    if concrete_name in generated:
                        new_decls.append(generated.pop(concrete_name))
        else:
            new_decls.append(d)

    return _flatten_refs_in_body(new_decls)


# ─────────────────────────────────────────────────────────────────────────────
# EXPRESSION TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

_NS_QUAL_RE = re.compile(r'\b[A-Za-z_]\w*\s*\.\s*([A-Z][a-zA-Z0-9_]*)')

def expr_to_py(expr: str) -> str:
    """Translate a hexpat expression string to valid Python."""
    if not expr:
        return expr
    # C logical operators already translated during parsing (&&→and etc)
    # $ → int(_dollar___offset)
    expr = re.sub(r'(?<![A-Za-z0-9_])\$(?![A-Za-z0-9_])', 'int(_dollar___offset)', expr)
    # Remove leftover namespace dots (e.g. agl . ImageFormat → ImageFormat)
    expr = _NS_QUAL_RE.sub(r'\1', expr)
    return expr.strip()


def offset_to_py(offset_expr: str) -> str:
    """Convert an offset expression to something usable after `@`.

    The right-hand side of `@` must be a Dollar (or IntStruct, which the
    __matmul__ guard converts).  Three cases:

    1. Pure numeric literal  → Dollar(n, byts)
    2. A field reference that holds an integer-like primitive (u32 etc.)
       → _resolve_offset(expr, byts) which calls int() to coerce it
    3. A field reference to a RelOffset struct
       → _resolve_offset(expr, byts) which calls relocate() on it
    """
    expr = expr_to_py(offset_expr)
    # Case 1: bare numeric literal
    if re.fullmatch(r'0x[0-9a-fA-F]+|\d+', expr):
        return f'Dollar({expr}, byts)'
    # Cases 2 & 3: delegate to the runtime helper emitted in the header
    return f'_resolve_offset({expr}, byts)'


# ─────────────────────────────────────────────────────────────────────────────
# CODE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_IND = '    '


class CodeGen:
    def __init__(self):
        self._lines: List[str] = []
        self._depth: int = 0

    def _w(self, line: str = '') -> None:
        prefix = _IND * self._depth
        self._lines.append(prefix + line if line else '')

    def _ind(self, n: int = 1) -> None:
        self._depth += n

    def _ded(self, n: int = 1) -> None:
        self._depth = max(0, self._depth - n)

    # ── Header ───────────────────────────────────────────────────────────────

    def _emit_header(self) -> None:
        self._w('from primitives import (')
        self._ind()
        self._w('Dollar, Struct, BitField, IntStruct,')
        self._w('u8, u16, u24, u32, u48, u64, u96, u128,')
        self._w('s8, s16, s24, s32, s48, s64, s96, s128,')
        self._w('Float, double, char, char16, Bool,')
        self._w('Padding, Array, Enum, sizeof, addressof,')
        self._ded()
        self._w(')')
        self._w()
        # Runtime helper: coerce any offset value to a Dollar.
        # Strategy: try int() first (covers plain ints and IntStruct/u32 primitives
        # that implement __int__). Only if that raises TypeError do we treat it as
        # a RelOffset-style struct where abs_pos = stored_read_pos + relative_value.
        self._w('def _resolve_offset(o, byts):')
        self._ind()
        self._w('from primitives import Dollar')
        self._w('if isinstance(o, Dollar): return o')
        self._w('try:')
        self._ind()
        self._w('return Dollar(int(o), byts)')
        self._ded()
        self._w('except TypeError:')
        self._ind()
        self._w('# RelOffset-style struct: abs pos = field read pos + relative value')
        self._w('val = int(o.value)')
        self._w('base = int(o.offset) if not callable(o.offset) else 0')
        self._w('abs_pos = (base + val) if val > 0 else 0')
        self._w('return Dollar(abs_pos, byts)')
        self._ded()
        self._ded()
        self._w()

    # ── Struct ───────────────────────────────────────────────────────────────

    def _emit_struct(self, s: StructDecl) -> None:
        self._w(f'class {s.name}(Struct):')
        self._ind()
        self._w('def __init__(self, name: str=""):')
        self._ind()
        self._w('super().__init__(name)')
        self._ded()
        self._w()
        self._w('def __matmul__(self, _dollar___offset):')
        self._ind()
        # Guard
        self._w('if not (isinstance(_dollar___offset, Dollar)'
                ' or isinstance(_dollar___offset, IntStruct)):')
        self._ind()
        self._w('raise Exception('
                'f\'An object of class "Dollar" must be used with the '
                '"@" operator. {type(_dollar___offset)} was used instead\')')
        self._ded()
        self._w('if isinstance(_dollar___offset, IntStruct):')
        self._ind()
        self._w('_dollar___offset = _dollar___offset.to_dollar()')
        self._ded()
        self._w('_dollar___offset_copy = _dollar___offset.copy()')
        self._w('byts = _dollar___offset.byts')
        self._emit_body(s.body)
        self._w('super().init_struct(_dollar___offset_copy, _dollar___offset.copy())')
        self._w('return self')
        self._ded()
        self._ded()
        self._w()

    # ── Body dispatch ────────────────────────────────────────────────────────

    def _emit_body(self, body: List) -> None:
        if not body:
            self._w('pass')
            return
        for node in body:
            self._emit_node(node)

    def _emit_node(self, node) -> None:
        if isinstance(node, SimpleField):
            self._emit_field(node)
        elif isinstance(node, PaddingField):
            self._emit_padding(node)
        elif isinstance(node, IfBlock):
            self._emit_if(node)
        elif isinstance(node, WhileBlock):
            self._emit_while(node)
        elif isinstance(node, BreakStmt):
            self._w('break')
        elif isinstance(node, StructDecl):
            pass  # nested structs collected and emitted at program level

    def _emit_field(self, f: SimpleField) -> None:
        py_type = f.type_ref.py_name()
        name    = f.name

        # `= $` initializer: capture current position, no bytes read
        if f.value_expr is not None:
            vexpr = f.value_expr.strip()
            if vexpr == '$' or vexpr == 'int ( _dollar___offset )':
                # Capture current position as an integer, don't advance
                self._w(f'{name} = int(_dollar___offset)')
                self._w(f'self.{name} = {name}')
                return
            else:
                # Read normally, then override value
                pass  # fall through to normal read, value_expr used as override

        offset_src = '_dollar___offset'
        if f.offset_expr:
            offset_src = offset_to_py(f.offset_expr)

        if f.is_auto_array:
            # Null-terminated / dynamic — special per type
            prim = f.type_ref.name
            if prim in ('char', 'char16'):
                step = 2 if prim == 'char16' else 1
                enc  = 'utf-16-le' if prim == 'char16' else 'utf-8'
                null = 'b"\\x00\\x00"' if prim == 'char16' else 'b"\\x00"'
                self._w(f'_cs_s = int(_dollar___offset)')
                self._w(f'_cs_e = _cs_s')
                self._w(f'while _cs_e + {step} <= len(byts) and byts[_cs_e:_cs_e+{step}] != {null}:')
                self._ind()
                self._w(f'_cs_e += {step}')
                self._ded()
                self._w(f'{name} = byts[_cs_s:_cs_e].decode("{enc}", errors="replace")')
                self._w(f'_dollar___offset += (_cs_e - _cs_s + {step})')
                self._w(f'self.{name} = {name}')
            else:
                self._w(f'{name} = None  # dynamic-length array not supported for {py_type}[]')
        if f.value_expr is not None:
            vexpr = f.value_expr.strip()
            if vexpr == '$' or vexpr == 'int ( _dollar___offset )':
                self._w(f'{name} = int(_dollar___offset)')
                self._w(f'self.{name} = {name}')
                return
            else:
                # Computed assignment — evaluate the expression, don't read from stream
                self._w(f'{name} = {expr_to_py(vexpr)}')
                self._w(f'self.{name} = {name}')
                return   # <-- add this return so it doesn't fall through to the normal read
            self._w(f'self.{name} = {name}')
        else:
            self._w(f'{name}: {py_type} = {py_type}(\'{name}\') @ {offset_src}')
            self._w(f'self.{name} = {name}')

    def _emit_padding(self, p: PaddingField) -> None:
        size = expr_to_py(p.size_expr)
        pname = f'_padding_{p.index}'
        self._w(f'self.{pname}: Padding = Padding({size}, 0, \'{pname}\') @ _dollar___offset')

    def _emit_if(self, node: IfBlock) -> None:
        cond = expr_to_py(node.cond_expr)
        self._w(f'if {cond}:')
        self._ind()
        self._emit_body(node.then_body)
        self._ded()
        if node.else_body:
            self._w('else:')
            self._ind()
            self._emit_body(node.else_body)
            self._ded()

    def _emit_while(self, node: WhileBlock) -> None:
        cond = expr_to_py(node.cond_expr)
        self._w(f'while {cond}:')
        self._ind()
        self._emit_body(node.body)
        self._ded()

    # ── Enum ─────────────────────────────────────────────────────────────────

    def _emit_enum(self, e: EnumDecl) -> None:
        base = PRIM_MAP.get(e.base_type, e.base_type)
        self._w(f'class {e.name}(Enum):')
        self._ind()

        pairs: List[str] = []
        auto_val = 0
        for (vname, vexpr) in e.values:
            if vexpr is not None:
                try:
                    v: object = int(vexpr, 0)
                    auto_val = int(v) + 1
                except (ValueError, TypeError):
                    v = expr_to_py(str(vexpr))
                    auto_val += 1
            else:
                v = auto_val
                auto_val += 1
            pairs.append(f'    {v!r}: {vname!r},')

        self._w('_enum__dict___ = {')
        for p in pairs:
            self._lines.append(_IND * self._depth + p)
        self._w('}')
        self._w(f'def __init__(self, value=None, name: str=""):')
        self._ind()
        self._w(f'super().__init__({base}, value, name)')
        self._ded()
        self._ded()
        self._w()

    # ── Bitfield ─────────────────────────────────────────────────────────────

    def _emit_bitfield(self, bf: BitfieldDecl) -> None:
        # Determine backing type from total bit count
        total = 0
        for (_, bits_str) in bf.fields:
            try:
                total += int(bits_str)
            except ValueError:
                total += 8
        backing = 'u8' if total <= 8 else ('u16' if total <= 16
                  else ('u32' if total <= 32 else 'u64'))

        self._w(f'class {bf.name}(Struct):')
        self._ind()
        self._w('def __init__(self, name: str=""):')
        self._ind()
        self._w('super().__init__(name)')
        self._ded()
        self._w()
        self._w('def __matmul__(self, _dollar___offset):')
        self._ind()
        self._w('if not (isinstance(_dollar___offset, Dollar)'
                ' or isinstance(_dollar___offset, IntStruct)):')
        self._ind()
        self._w('raise Exception('
                'f\'An object of class "Dollar" must be used with the '
                '"@" operator. {type(_dollar___offset)} was used instead\')')
        self._ded()
        self._w('if isinstance(_dollar___offset, IntStruct):')
        self._ind()
        self._w('_dollar___offset = _dollar___offset.to_dollar()')
        self._ded()
        self._w('_dollar___offset_copy = _dollar___offset.copy()')
        self._w(f'_raw = {backing}(\'_raw\') @ _dollar___offset')
        self._w('_val = int(_raw)')
        self._w('_bit_pos = 0')

        for (fname, bits_str) in bf.fields:
            bits_str = bits_str.strip()
            try:
                bits_int = int(bits_str)
                mask = (1 << bits_int) - 1
                self._w(f'{fname} = (_val >> _bit_pos) & 0x{mask:X}')
                self._w(f'self.{fname} = {fname}')
                self._w(f'_bit_pos += {bits_int}')
            except ValueError:
                self._w(f'# unresolved bit count {bits_str!r} for {fname!r}')

        self._w('super().init_struct(_dollar___offset_copy, _dollar___offset.copy())')
        self._w('return self')
        self._ded()
        self._ded()
        self._w()

    # ── Using / Placement / Function ─────────────────────────────────────────

    def _emit_using(self, u: UsingDecl) -> None:
        if u.target:
            self._w(f'{u.alias} = {u.target}')

    def _emit_placement(self, p: TopLevelPlacement) -> None:
        py_type = p.type_ref.py_name()
        if p.offset_expr:
            off = offset_to_py(p.offset_expr)
            self._w(f'{p.name} = {py_type}(\'{p.name}\') @ {off}')
        else:
            self._w(f'{p.name} = {py_type}(\'{p.name}\') @ _dollar___offset')

    # ── Program ──────────────────────────────────────────────────────────────

    def _collect_nested(self, decls: List) -> List:
        """Flatten nested StructDecls to top-level, preserving declaration order."""
        result: List = []
        for d in decls:
            if isinstance(d, StructDecl) and not d.template_params:
                # Emit nested struct/enum/bitfield first
                nested = [n for n in d.body
                          if isinstance(n, (StructDecl, EnumDecl, BitfieldDecl))]
                result.extend(self._collect_nested(nested))
                result.append(d)
            else:
                result.append(d)
        return result

    def emit_program(self, prog: Program) -> str:
        self._emit_header()
        self._w('_dollar___offset = Dollar(0x00, byts)')
        self._w()

        flat = self._collect_nested(prog.decls)

        for d in flat:
            if isinstance(d, StructDecl) and not d.template_params:
                self._emit_struct(d)
            elif isinstance(d, EnumDecl):
                self._emit_enum(d)
            elif isinstance(d, BitfieldDecl):
                self._emit_bitfield(d)
            elif isinstance(d, UsingDecl):
                self._emit_using(d)
            elif isinstance(d, TopLevelPlacement):
                self._emit_placement(d)
            elif isinstance(d, FunctionDecl):
                pass  # intentionally skipped

        return '\n'.join(self._lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def compile_text(src: str) -> str:
    """
    Compile a hexpat source string into Python code that uses primitives.py.
    The caller is responsible for injecting `byts` (the file bytes) into the
    exec environment before running the output.
    """
    tokens  = Lexer(src).tokenize()
    program = Parser(tokens).parse()
    program.decls = flatten_namespaces(program.decls)
    program.decls = resolve_templates(program.decls)
    return CodeGen().emit_program(program)
