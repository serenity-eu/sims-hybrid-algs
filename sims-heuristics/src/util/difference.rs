pub struct Difference<I>
where
    I: Iterator,
{
    iter1: std::iter::Peekable<I>,
    iter2: std::iter::Peekable<I>,
}

impl<I, T> Iterator for Difference<I>
where
    I: Iterator<Item = T>,
    T: Ord + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(&next1), Some(&next2)) = (self.iter1.peek(), self.iter2.peek()) {
            match next1.cmp(&next2) {
                std::cmp::Ordering::Less => return self.iter1.next(),
                std::cmp::Ordering::Greater => {
                    self.iter2.next();
                }
                std::cmp::Ordering::Equal => {
                    self.iter1.next();
                    self.iter2.next();
                }
            }
        }
        self.iter1.next()
    }
}

impl<I> Difference<I>
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

pub trait DifferenceIterator<T>: Iterator<Item = T> + Sized {
    fn difference(self, other: Self) -> Difference<Self> {
        Difference::new(self, other)
    }
}

impl<T, I: Iterator<Item = T> + Sized> DifferenceIterator<T> for I {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference_empty_iterators() {
        let iter1: Vec<i32> = vec![];
        let iter2: Vec<i32> = vec![];

        let diff = Difference::new(iter1.into_iter(), iter2.into_iter());
        assert_eq!(diff.collect::<Vec<i32>>(), Vec::<i32>::new());
    }

    #[test]
    fn test_difference_no_common_elements() {
        let iter1 = vec![1, 2, 3];
        let iter2 = vec![4, 5, 6];

        let diff = Difference::new(iter1.into_iter(), iter2.into_iter());
        assert_eq!(diff.collect::<Vec<i32>>(), vec![1, 2, 3]);
    }

    #[test]
    fn test_difference_some_common_elements() {
        let iter1 = vec![1, 2, 3, 4, 5];
        let iter2 = vec![3, 4, 5, 6, 7];

        let diff = Difference::new(iter1.into_iter(), iter2.into_iter());
        assert_eq!(diff.collect::<Vec<i32>>(), vec![1, 2]);
    }

    #[test]
    fn test_difference_all_common_elements() {
        let iter1 = vec![1, 2, 3];
        let iter2 = vec![1, 2, 3];

        let diff = Difference::new(iter1.into_iter(), iter2.into_iter());
        assert_eq!(diff.collect::<Vec<i32>>(), Vec::<i32>::new());
    }
}
