mod fixed_vec;
pub use fixed_vec::{FixedVec, IntoIter};

#[cfg(test)]
mod tests {
    use crate::fixed_vec::FixedVec;

    #[test]
    fn single_thread() {
        let vec = FixedVec::<u64>::new(2);
        assert_eq!(Ok(0), vec.push(1));
        assert_eq!(Ok(1), vec.push(2));
        // No more space, the value should be returned.
        assert_eq!(Err(4), vec.push(4));
        // This should be in bounds.
        assert_eq!(Some(&2u64), vec.get(1));
        // This should be out of bounds.
        assert_eq!(None, vec.get(2));
        println!("{:?}", vec);
        for item in vec.clone() {
            println!("{}", item);
        }
        for item in vec.into_iter().rev() {
            println!("{}", item);
        }
    }

    #[test]
    fn realloc() {
        let mut vec = FixedVec::<String>::new(2);
        vec.push("a".to_string()).unwrap();
        vec.push("b".to_string()).unwrap();
        vec.realloc(4);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.capacity(), 4);
        assert_eq!(vec[0], "a");
        assert_eq!(vec[1], "b");
    }

    #[test]
    fn concurrent_push() {
        use std::sync::Arc;
        use std::thread;

        let vec = Arc::new(FixedVec::new(1000));
        let mut handles = vec![];

        for t in 0..10 {
            let vec = Arc::clone(&vec);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    vec.push(t * 100 + i).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(vec.len(), 1000);
        let mut sum = 0;
        for &val in vec.iter() {
            sum += val;
        }
        // Sum of 0..999 is (999 * 1000) / 2 = 499500
        assert_eq!(sum, 499500);
    }
}
