pub mod add;
pub mod multiply;
// TODO: add divide module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add::add(2.0, 3.0), 5.0);
        assert_eq!(add::add(-1.0, 1.0), 0.0);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply::multiply(3.0, 4.0), 12.0);
        assert_eq!(multiply::multiply(0.0, 100.0), 0.0);
    }

    // Uncomment these tests after implementing divide:
    //
    // #[test]
    // fn test_divide() {
    //     assert_eq!(divide::divide(10.0, 2.0), Ok(5.0));
    //     assert_eq!(divide::divide(7.0, 2.0), Ok(3.5));
    // }
    //
    // #[test]
    // fn test_divide_by_zero() {
    //     assert_eq!(divide::divide(10.0, 0.0), Err("division by zero".to_string()));
    // }
}
