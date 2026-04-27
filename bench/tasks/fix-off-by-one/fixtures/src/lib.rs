/// Paginate items into pages of a given size.
/// Returns the total number of pages needed.
pub fn paginate(total_items: usize, page_size: usize) -> usize {
    if page_size == 0 {
        return 0;
    }
    // BUG: this returns one too many pages when total_items is exactly
    // divisible by page_size.
    total_items / page_size + 1
}

/// Get the items for a specific page (0-indexed).
/// Returns (start_index, end_index_exclusive).
pub fn page_range(total_items: usize, page_size: usize, page: usize) -> (usize, usize) {
    let start = page * page_size;
    let end = (start + page_size).min(total_items);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_division() {
        // 10 items, 5 per page → exactly 2 pages
        assert_eq!(paginate(10, 5), 2);
    }

    #[test]
    fn test_remainder() {
        // 11 items, 5 per page → 3 pages (last page has 1 item)
        assert_eq!(paginate(11, 5), 3);
    }

    #[test]
    fn test_single_page() {
        assert_eq!(paginate(3, 10), 1);
    }

    #[test]
    fn test_zero_items() {
        assert_eq!(paginate(0, 5), 0);
    }

    #[test]
    fn test_one_item() {
        assert_eq!(paginate(1, 5), 1);
    }

    #[test]
    fn test_page_size_one() {
        assert_eq!(paginate(5, 1), 5);
    }

    #[test]
    fn test_zero_page_size() {
        assert_eq!(paginate(10, 0), 0);
    }

    #[test]
    fn test_page_range() {
        assert_eq!(page_range(11, 5, 0), (0, 5));
        assert_eq!(page_range(11, 5, 1), (5, 10));
        assert_eq!(page_range(11, 5, 2), (10, 11));
    }
}
