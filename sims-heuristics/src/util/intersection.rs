pub struct Intersection<I>
where
    I: Iterator,
{
    iter1: std::iter::Peekable<I>,
    iter2: std::iter::Peekable<I>,
}

impl<I, T> Iterator for Intersection<I>
where
    I: Iterator<Item = T>,
    T: Ord + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(&next1), Some(&next2)) = (self.iter1.peek(), self.iter2.peek()) {
            match next1.cmp(&next2) {
                std::cmp::Ordering::Less => {
                    self.iter1.next();
                }
                std::cmp::Ordering::Greater => {
                    self.iter2.next();
                }
                std::cmp::Ordering::Equal => {
                    return Some(self.iter1.next().unwrap());
                }
            }
        }
        None
    }
}

impl<I> Intersection<I>
where
    I: Iterator,
{
    fn new(iter1: I, iter2: I) -> Self {
        Intersection {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
        }
    }
}

pub trait IntersectionIterator<T>: Iterator<Item = T> + Sized {
    fn intersection(self, other: Self) -> Intersection<Self> {
        Intersection::new(self, other)
    }
}

impl<T, I: Iterator<Item = T> + Sized> IntersectionIterator<T> for I {}
