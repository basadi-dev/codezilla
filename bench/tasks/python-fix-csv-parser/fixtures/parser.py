"""Simple CSV parser with support for quoted fields and custom delimiters."""


def parse_csv(text: str, delimiter: str = ",", quote_char: str = '"') -> list[list[str]]:
    """Parse CSV text into a list of rows, where each row is a list of fields.

    Supports:
    - Custom delimiters
    - Quoted fields (fields wrapped in quote_char)
    - Escaped quotes inside quoted fields (doubled quote_char)
    - Newlines inside quoted fields
    - Leading/trailing whitespace is preserved inside quoted fields
      but stripped from unquoted fields
    """
    rows = []
    current_row = []
    current_field = []
    in_quotes = False
    i = 0

    while i < len(text):
        char = text[i]

        if in_quotes:
            if char == quote_char:
                # Check for escaped quote (doubled quote char)
                if i + 1 < len(text) and text[i + 1] == quote_char:
                    current_field.append(quote_char)
                    i += 2
                    continue
                else:
                    # End of quoted field
                    in_quotes = False
                    i += 1
                    continue
            else:
                current_field.append(char)
                i += 1
                continue

        # Not in quotes
        if char == quote_char:
            in_quotes = True
            i += 1
            continue

        if char == delimiter:
            # BUG 1: should strip unquoted fields, but strips quoted ones too
            current_row.append("".join(current_field))
            current_field = []
            i += 1
            continue

        if char == "\n":
            current_row.append("".join(current_field))
            current_field = []
            # BUG 2: appends empty rows for trailing newlines
            rows.append(current_row)
            current_row = []
            i += 1
            continue

        if char == "\r":
            # BUG 3: doesn't handle \r\n properly — treats \r and \n as separate newlines
            current_row.append("".join(current_field))
            current_field = []
            rows.append(current_row)
            current_row = []
            i += 1
            continue

        current_field.append(char)
        i += 1

    # Don't forget the last field/row
    if current_field or current_row:
        current_row.append("".join(current_field))
        rows.append(current_row)

    return rows


def to_csv(rows: list[list[str]], delimiter: str = ",", quote_char: str = '"') -> str:
    """Convert a list of rows back to CSV text.

    Fields containing the delimiter, quote_char, or newlines are quoted.
    Quote chars inside fields are doubled.
    """
    lines = []
    for row in rows:
        fields = []
        for field in row:
            needs_quoting = (
                delimiter in field or quote_char in field or "\n" in field
            )
            if needs_quoting:
                escaped = field.replace(quote_char, quote_char + quote_char)
                fields.append(f"{quote_char}{escaped}{quote_char}")
            else:
                fields.append(field)
        lines.append(delimiter.join(fields))
    return "\n".join(lines)
