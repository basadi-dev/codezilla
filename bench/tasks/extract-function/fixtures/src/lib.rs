#[derive(Debug, Clone)]
pub struct Record {
    pub name: String,
    pub age: i32,
    pub email: String,
}

pub fn process_records(records: &[Record]) -> Vec<String> {
    let mut results = Vec::new();

    for record in records {
        // --- Validation section (should be extracted) ---
        if record.name.is_empty() {
            results.push(format!("Invalid record: empty name"));
            continue;
        }
        if record.age < 0 {
            results.push(format!("Invalid record '{}': negative age", record.name));
            continue;
        }
        if !record.email.contains('@') {
            results.push(format!(
                "Invalid record '{}': invalid email '{}'",
                record.name, record.email
            ));
            continue;
        }
        // --- End validation section ---

        // Processing
        let display = format!(
            "{} (age {}) <{}>",
            record.name, record.age, record.email
        );
        results.push(display);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(name: &str, age: i32, email: &str) -> Record {
        Record {
            name: name.to_string(),
            age,
            email: email.to_string(),
        }
    }

    #[test]
    fn test_valid_records() {
        let records = vec![
            make_record("Alice", 30, "alice@example.com"),
            make_record("Bob", 25, "bob@example.com"),
        ];
        let results = process_records(&records);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("Alice"));
        assert!(results[1].contains("Bob"));
    }

    #[test]
    fn test_empty_name() {
        let records = vec![make_record("", 30, "test@example.com")];
        let results = process_records(&records);
        assert!(results[0].contains("empty name"));
    }

    #[test]
    fn test_negative_age() {
        let records = vec![make_record("Charlie", -5, "charlie@example.com")];
        let results = process_records(&records);
        assert!(results[0].contains("negative age"));
    }

    #[test]
    fn test_invalid_email() {
        let records = vec![make_record("Dave", 40, "not-an-email")];
        let results = process_records(&records);
        assert!(results[0].contains("invalid email"));
    }

    #[test]
    fn test_mixed() {
        let records = vec![
            make_record("Eve", 28, "eve@example.com"),
            make_record("", 30, "nobody@example.com"),
            make_record("Frank", -1, "frank@example.com"),
            make_record("Grace", 35, "bad-email"),
            make_record("Heidi", 22, "heidi@example.com"),
        ];
        let results = process_records(&records);
        assert_eq!(results.len(), 5);
        assert!(results[0].contains("Eve"));
        assert!(results[1].contains("empty name"));
        assert!(results[2].contains("negative age"));
        assert!(results[3].contains("invalid email"));
        assert!(results[4].contains("Heidi"));
    }

    #[test]
    fn test_validate_record_exists() {
        // This test ensures the validate_record function was actually created.
        let good = make_record("Test", 25, "test@example.com");
        assert!(validate_record(&good).is_ok());

        let bad_name = make_record("", 25, "test@example.com");
        assert!(validate_record(&bad_name).is_err());

        let bad_age = make_record("Test", -1, "test@example.com");
        assert!(validate_record(&bad_age).is_err());

        let bad_email = make_record("Test", 25, "no-at-sign");
        assert!(validate_record(&bad_email).is_err());
    }
}
