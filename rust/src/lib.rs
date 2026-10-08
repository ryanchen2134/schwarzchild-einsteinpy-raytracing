//! Schwarzschild null-geodesic ray tracer.
//!
//! Pipeline: [`camera`] builds pixel rays → [`initial_conditions`] turns each ray
//! into a null covariant 4-momentum → [`integrator`] follows it with the FANTASY
//! order-2 symplectic scheme → [`shading`] classifies the exit and samples the
//! sky through a [`sky_patch::SkyPatch`] → [`output`] writes the image and CSVs.
//! [`render`] wires the stages together; [`flat`] is the no-gravity reference.
//!
//! Units: G = c = 1. Lengths are in units of the black-hole mass `M`.
//! Momenta are *covariant* components `p_μ`; the Hamiltonian is
//! `H = ½ g^{μν} p_μ p_ν`, so `dq^μ/dλ = g^{μν} p_ν`.

pub mod camera;
pub mod coords;
pub mod flat;
pub mod initial_conditions;
pub mod integrator;
pub mod output;
pub mod render;
pub mod sampling;
pub mod schwarzschild;
pub mod shading;
pub mod sky_patch;
pub mod texture;
