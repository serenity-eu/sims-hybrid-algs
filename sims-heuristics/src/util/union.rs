pub struct Union<I>
where
    I: Iterator,
{
    iter1: std::iter::Peekable<I>,
    iter2: std::iter::Peekable<I>,
}

impl<I, T> Iterator for Union<I>
where
    I: Iterator<Item = T>,
    T: Ord + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.iter1.peek().copied(), self.iter2.peek().copied()) {
            (None, None) => None,
            (Some(_), None) => self.iter1.next(),
            (None, Some(_)) => self.iter2.next(),
            (Some(next1), Some(next2)) => match next1.cmp(&next2) {
                std::cmp::Ordering::Less => self.iter1.next(),
                std::cmp::Ordering::Greater => self.iter2.next(),
                std::cmp::Ordering::Equal => {
                    // Emit only once, advance both.
                    let item = self.iter1.next();
                    self.iter2.next();
                    item
                }
            },
        }
    }
}

impl<I> Union<I>
where
    I: Iterator,
{
    fn new(iter1: I, iter2: I) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
        }
    }
}

pub trait UnionIterator<T>: Iterator<Item = T> + Sized {
    fn union(self, other: Self) -> Union<Self> {
        Union::new(self, other)
    }
}

impl<T, I: Iterator<Item = T> + Sized> UnionIterator<T> for I {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_empty_iterators() {
        let iter1: Vec<i32> = vec![];
        let iter2: Vec<i32> = vec![];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), Vec::<i32>::new());
    }

    #[test]
    fn test_union_left_only() {
        let iter1 = vec![1, 2, 3];
        let iter2: Vec<i32> = vec![];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), vec![1, 2, 3]);
    }

    #[test]
    fn test_union_right_only() {
        let iter1: Vec<i32> = vec![];
        let iter2 = vec![4, 5, 6];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), vec![4, 5, 6]);
    }

    #[test]
    fn test_union_no_common_elements() {
        let iter1 = vec![1, 2, 3];
        let iter2 = vec![4, 5, 6];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_union_some_common_elements() {
        let iter1 = vec![1, 2, 3, 4, 5];
        let iter2 = vec![3, 4, 5, 6, 7];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_union_all_common_elements() {
        let iter1 = vec![1, 2, 3];
        let iter2 = vec![1, 2, 3];

        let union = iter1.into_iter().union(iter2.into_iter());
        assert_eq!(union.collect::<Vec<i32>>(), vec![1, 2, 3]);
    }
}
