/// Application configuration.
pub struct Config {
    pub name: String,
    pub port: u16,
    pub debug: bool,
    pub max_connections: usize,
    pub log_level: String,
}

impl Config {
    pub fn default_config() -> Self {
        Config {
            name: "my-app".to_string(),
            port: 8080,
            debug: false,
            max_connections: 100,
            log_level: "info".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let config = Config::default_config();
        let json = config.to_json();
        let parsed = Config::from_json(&json).unwrap();
        assert_eq!(parsed.name, "my-app");
        assert_eq!(parsed.port, 8080);
        assert_eq!(parsed.debug, false);
        assert_eq!(parsed.max_connections, 100);
        assert_eq!(parsed.log_level, "info");
    }

    #[test]
    fn test_json_contains_fields() {
        let config = Config::default_config();
        let json = config.to_json();
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"port\""));
        assert!(json.contains("\"debug\""));
        assert!(json.contains("8080"));
    }

    #[test]
    fn test_from_custom_json() {
        let json = r#"{"name":"test","port":3000,"debug":true,"max_connections":50,"log_level":"debug"}"#;
        let config = Config::from_json(json).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.port, 3000);
        assert_eq!(config.debug, true);
    }
}
