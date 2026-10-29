#include "core/json.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>

#include <xlocale.h>

namespace core::json {

Value Value::boolean(bool b) {
    Value v;
    v.type_ = Type::Bool;
    v.bool_ = b;
    return v;
}

Value Value::number(double d) {
    Value v;
    v.type_ = Type::Number;
    v.number_ = d;
    return v;
}

Value Value::integer(int64_t i) {
    Value v;
    v.type_ = Type::Number;
    v.number_ = double(i);
    v.exactInt_ = true;
    v.negative_ = i < 0;
    v.magnitude_ = i < 0 ? uint64_t(0) - uint64_t(i) : uint64_t(i);
    return v;
}

Value Value::unsignedInteger(uint64_t u) {
    Value v;
    v.type_ = Type::Number;
    v.number_ = double(u);
    v.exactInt_ = true;
    v.magnitude_ = u;
    return v;
}

Value Value::string(std::string s) {
    Value v;
    v.type_ = Type::String;
    v.string_ = std::move(s);
    return v;
}

Value Value::array() {
    Value v;
    v.type_ = Type::Array;
    return v;
}

Value Value::object() {
    Value v;
    v.type_ = Type::Object;
    return v;
}

bool Value::getUint64(uint64_t& out) const {
    if (type_ != Type::Number || !exactInt_ || negative_) return false;
    out = magnitude_;
    return true;
}

bool Value::getInt64(int64_t& out) const {
    if (type_ != Type::Number || !exactInt_) return false;
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) + (negative_ ? 1 : 0);
    if (magnitude_ > limit) return false;
    out = negative_ ? int64_t(uint64_t(0) - magnitude_) : int64_t(magnitude_);
    return true;
}

const Value* Value::find(std::string_view key) const {
    for (const auto& [k, v] : members_)
        if (k == key) return &v;
    return nullptr;
}

Value& Value::push(Value v) {
    items_.push_back(std::move(v));
    return items_.back();
}

Value& Value::set(std::string key, Value v) {
    for (auto& [k, existing] : members_) {
        if (k == key) {
            existing = std::move(v);
            return existing;
        }
    }
    members_.emplace_back(std::move(key), std::move(v));
    return members_.back().second;
}

bool Value::operator==(const Value& o) const {
    if (type_ != o.type_) return false;
    switch (type_) {
        case Type::Null: return true;
        case Type::Bool: return bool_ == o.bool_;
        case Type::Number:
            if (exactInt_ && o.exactInt_) return negative_ == o.negative_ && magnitude_ == o.magnitude_;
            return number_ == o.number_;
        case Type::String: return string_ == o.string_;
        case Type::Array: return items_ == o.items_;
        case Type::Object: return members_ == o.members_;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Parser

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    ParseResult run() {
        ParseResult r;
        skipWs();
        if (!parseValue(r.value, 0)) {
            r.error = error_;
            return r;
        }
        skipWs();
        if (pos_ != s_.size()) {
            fail("unexpected characters after the document");
            r.error = error_;
            return r;
        }
        r.ok = true;
        return r;
    }

private:
    static constexpr int kMaxDepth = 64;

    std::string_view s_;
    size_t pos_ = 0;
    std::string error_;

    bool atEnd() const { return pos_ >= s_.size(); }
    char peek() const { return s_[pos_]; }
    static bool isDigit(char c) { return c >= '0' && c <= '9'; }

    bool fail(const char* msg) {
        size_t line = 1, col = 1;
        for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
            if (s_[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        error_ = "line " + std::to_string(line) + ", column " + std::to_string(col) + ": " + msg;
        return false;
    }

    void skipWs() {
        while (!atEnd() && (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r')) ++pos_;
    }

    bool parseValue(Value& out, int depth) {
        if (depth > kMaxDepth) return fail("nesting too deep");
        if (atEnd()) return fail("unexpected end of input");
        switch (peek()) {
            case '{': return parseObject(out, depth);
            case '[': return parseArray(out, depth);
            case '"': {
                std::string s;
                if (!parseString(s)) return false;
                out = Value::string(std::move(s));
                return true;
            }
            case 't': return parseLiteral("true", Value::boolean(true), out);
            case 'f': return parseLiteral("false", Value::boolean(false), out);
            case 'n': return parseLiteral("null", Value(), out);
            default:
                if (peek() == '-' || isDigit(peek())) return parseNumber(out);
                return fail("unexpected character");
        }
    }

    bool parseLiteral(std::string_view word, Value v, Value& out) {
        if (s_.substr(pos_, word.size()) != word) return fail("invalid literal");
        pos_ += word.size();
        out = std::move(v);
        return true;
    }

    bool parseNumber(Value& out) {
        const size_t start = pos_;
        bool negative = false, integral = true;
        if (peek() == '-') {
            negative = true;
            ++pos_;
        }
        if (atEnd()) return fail("invalid number");
        if (peek() == '0') {
            ++pos_;
        } else if (isDigit(peek())) {
            while (!atEnd() && isDigit(peek())) ++pos_;
        } else {
            return fail("invalid number");
        }
        if (!atEnd() && peek() == '.') {
            integral = false;
            ++pos_;
            if (atEnd() || !isDigit(peek())) return fail("invalid number");
            while (!atEnd() && isDigit(peek())) ++pos_;
        }
        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            integral = false;
            ++pos_;
            if (!atEnd() && (peek() == '+' || peek() == '-')) ++pos_;
            if (atEnd() || !isDigit(peek())) return fail("invalid number");
            while (!atEnd() && isDigit(peek())) ++pos_;
        }
        const char* first = s_.data() + start;
        const char* last = s_.data() + pos_;
        // strtod_l with the C locale: std::from_chars(double) needs macOS 26, and
        // plain strtod would follow the process locale's decimal separator.
        const std::string literal(first, last);
        char* end = nullptr;
        const double d = strtod_l(literal.c_str(), &end, LC_C_LOCALE);
        if (end != literal.c_str() + literal.size() || !std::isfinite(d)) return fail("number out of range");
        out = Value::number(d);
        if (integral) {
            uint64_t mag = 0;
            auto ri = std::from_chars(first + (negative ? 1 : 0), last, mag);
            if (ri.ec == std::errc() && ri.ptr == last) {
                out.exactInt_ = true;
                out.negative_ = negative && mag != 0;
                out.magnitude_ = mag;
            }
        }
        return true;
    }

    bool hex4(uint32_t& out) {
        if (s_.size() - pos_ < 4) return fail("truncated \\u escape");
        out = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') out |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= uint32_t(c - 'A' + 10);
            else return fail("invalid \\u escape");
        }
        return true;
    }

    static void appendUtf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(char(cp));
        } else if (cp < 0x800) {
            s.push_back(char(0xC0 | (cp >> 6)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(char(0xE0 | (cp >> 12)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(char(0xF0 | (cp >> 18)));
            s.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(char(0x80 | (cp & 0x3F)));
        }
    }

    bool parseString(std::string& out) {
        ++pos_;  // opening quote
        while (true) {
            if (atEnd()) return fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (atEnd()) return fail("unterminated escape");
            switch (s_[pos_++]) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (s_.substr(pos_, 2) != "\\u") return fail("unpaired surrogate");
                        pos_ += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("unpaired surrogate");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: return fail("invalid escape");
            }
        }
    }

    bool parseArray(Value& out, int depth) {
        ++pos_;
        out = Value::array();
        skipWs();
        if (!atEnd() && peek() == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            skipWs();
            Value v;
            if (!parseValue(v, depth + 1)) return false;
            out.items_.push_back(std::move(v));
            skipWs();
            if (atEnd()) return fail("unterminated array");
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == ']') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    bool parseObject(Value& out, int depth) {
        ++pos_;
        out = Value::object();
        skipWs();
        if (!atEnd() && peek() == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            skipWs();
            if (atEnd() || peek() != '"') return fail("expected a string key");
            std::string key;
            if (!parseString(key)) return false;
            if (out.find(key)) return fail(("duplicate key \"" + key + "\"").c_str());
            skipWs();
            if (atEnd() || peek() != ':') return fail("expected ':'");
            ++pos_;
            skipWs();
            Value v;
            if (!parseValue(v, depth + 1)) return false;
            out.members_.emplace_back(std::move(key), std::move(v));
            skipWs();
            if (atEnd()) return fail("unterminated object");
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            if (peek() == '}') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }
};

ParseResult parse(std::string_view text) { return Parser(text).run(); }

// ---------------------------------------------------------------------------
// Writer

namespace {

void dumpString(const std::string& s, std::string& out) {
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void dumpNumber(const Value& v, std::string& out) {
    uint64_t u = 0;
    int64_t i = 0;
    if (v.getUint64(u)) {
        out += std::to_string(u);
    } else if (v.getInt64(i)) {
        out += std::to_string(i);
    } else if (!std::isfinite(v.asDouble())) {
        out += "null";
    } else {
        char buf[32];
        auto r = std::to_chars(buf, buf + sizeof buf, v.asDouble());  // shortest round-trip form
        out.append(buf, r.ptr);
    }
}

bool isScalar(const Value& v) { return v.type() != Type::Array && v.type() != Type::Object; }

void newline(std::string& out, int indent, int level) {
    if (indent <= 0) return;
    out.push_back('\n');
    out.append(size_t(indent * level), ' ');
}

void dumpValue(const Value& v, std::string& out, int indent, int level) {
    switch (v.type()) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += v.asBool() ? "true" : "false"; break;
        case Type::Number: dumpNumber(v, out); break;
        case Type::String: dumpString(v.asString(), out); break;
        case Type::Array: {
            if (v.items().empty()) {
                out += "[]";
                break;
            }
            bool inlineItems = true;
            for (const auto& item : v.items()) inlineItems = inlineItems && isScalar(item);
            out.push_back('[');
            for (size_t k = 0; k < v.items().size(); ++k) {
                if (k) out += inlineItems && indent > 0 ? ", " : ",";
                if (!inlineItems) newline(out, indent, level + 1);
                dumpValue(v.items()[k], out, indent, level + 1);
            }
            if (!inlineItems) newline(out, indent, level);
            out.push_back(']');
            break;
        }
        case Type::Object: {
            if (v.members().empty()) {
                out += "{}";
                break;
            }
            out.push_back('{');
            for (size_t k = 0; k < v.members().size(); ++k) {
                if (k) out.push_back(',');
                newline(out, indent, level + 1);
                dumpString(v.members()[k].first, out);
                out += indent > 0 ? ": " : ":";
                dumpValue(v.members()[k].second, out, indent, level + 1);
            }
            newline(out, indent, level);
            out.push_back('}');
            break;
        }
    }
}

}  // namespace

std::string dump(const Value& v, int indent) {
    std::string out;
    dumpValue(v, out, indent, 0);
    if (indent > 0) out.push_back('\n');
    return out;
}

}  // namespace core::json
