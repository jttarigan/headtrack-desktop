// Minimal JSON reader and writer for config files and logs (no third-party
// dependency). Objects keep insertion order so dumps are deterministic; integer
// literals are kept exact so 64-bit seeds survive a round trip.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::json {

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
public:
    Value() = default;  // null
    static Value boolean(bool b);
    static Value number(double d);
    static Value integer(int64_t i);
    static Value unsignedInteger(uint64_t u);
    static Value string(std::string s);
    static Value array();
    static Value object();

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }

    bool asBool() const { return bool_; }
    double asDouble() const { return number_; }
    // True if the value is an integer literal that fits the target type.
    bool getUint64(uint64_t& out) const;
    bool getInt64(int64_t& out) const;
    const std::string& asString() const { return string_; }

    const std::vector<Value>& items() const { return items_; }
    const std::vector<std::pair<std::string, Value>>& members() const { return members_; }
    const Value* find(std::string_view key) const;

    Value& push(Value v);                  // array
    Value& set(std::string key, Value v);  // object; replaces an existing key

    bool operator==(const Value& other) const;

private:
    friend class Parser;

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    bool exactInt_ = false;  // literal was an integer within 64 bits
    bool negative_ = false;
    uint64_t magnitude_ = 0;
    std::string string_;
    std::vector<Value> items_;
    std::vector<std::pair<std::string, Value>> members_;
};

struct ParseResult {
    bool ok = false;
    Value value;
    std::string error;  // "line L, column C: message"
};

// Strict RFC 8259 parse; duplicate object keys are an error.
ParseResult parse(std::string_view text);

// indent > 0 pretty-prints; arrays of scalars stay on one line.
std::string dump(const Value& v, int indent = 2);

}  // namespace core::json
