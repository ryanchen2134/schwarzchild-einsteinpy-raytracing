//! Choice of the diagnostic rays whose full trajectories are recorded.

use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// `n` distinct flat pixel indices out of `n_pixels`, sorted. A `seed` makes the
/// choice reproducible.
pub fn choose_sample_pixels(seed: Option<u64>, n_pixels: usize, n: usize) -> Result<Vec<usize>> {
    if n > n_pixels {
        bail!("cannot sample {n} distinct pixels from {n_pixels}");
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut idx = rand::seq::index::sample(&mut rng, n_pixels, n).into_vec();
    idx.sort_unstable();
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_sampling_is_reproducible_and_distinct() {
        let a = choose_sample_pixels(Some(42), 100, 20).unwrap();
        let b = choose_sample_pixels(Some(42), 100, 20).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 20);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert!(choose_sample_pixels(Some(1), 10, 11).is_err());
        assert!(choose_sample_pixels(Some(1), 10, 0).unwrap().is_empty());
    }
}
