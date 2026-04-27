// TODO: Implement the Matrix struct and all methods described in the task prompt.
// The tests below define the exact API you must implement.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zero_matrix() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(1, 2), 0.0);
    }

    #[test]
    fn test_from_vec() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(2, 2), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
    }

    #[test]
    fn test_set_and_get() {
        let mut m = Matrix::new(2, 2);
        m.set(0, 1, 42.0);
        assert_eq!(m.get(0, 1), 42.0);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_scale() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let s = m.scale(2.0);
        assert_eq!(s.get(0, 0), 2.0);
        assert_eq!(s.get(0, 1), 4.0);
        assert_eq!(s.get(0, 2), 6.0);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(3, 2);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_multiply() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.multiply(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(0, 0), 58.0);  // 1*7 + 2*9 + 3*11
        assert_eq!(c.get(0, 1), 64.0);  // 1*8 + 2*10 + 3*12
        assert_eq!(c.get(1, 0), 139.0); // 4*7 + 5*9 + 6*11
        assert_eq!(c.get(1, 1), 154.0); // 4*8 + 5*10 + 6*12
    }

    #[test]
    fn test_multiply_dimension_mismatch() {
        let a = Matrix::new(2, 3);
        let b = Matrix::new(2, 3);
        assert!(a.multiply(&b).is_err());
    }

    #[test]
    fn test_identity_multiply() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let i = Matrix::identity(2);
        let result = a.multiply(&i).unwrap();
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 2.0);
        assert_eq!(result.get(1, 0), 3.0);
        assert_eq!(result.get(1, 1), 4.0);
    }

    #[test]
    fn test_display() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let display = format!("{}", m);
        assert!(display.contains("1"));
        assert!(display.contains("6"));
        // Should have at least 2 lines
        assert!(display.lines().count() >= 2);
    }
}
