//! # Game of Life Neural Network Simulation
//!
//! This is the main entry point for a neural network-based Conway's Game of Life implementation
//! using the Burn deep learning framework. The program simulates cellular automata by:
//!
//! - Loading initial patterns from text files
//! - Using a convolutional neural network to compute next generations
//! - Outputting the evolution in real-time to stdout
//! - Supporting both finite and infinite iteration modes
//!
//! ## Architecture
//!
//! The system consists of several key components:
//! - **args**: Command line argument parsing and file input handling
//! - **model**: Neural network model implementing Game of Life rules via convolution
//! - **inference**: Output formatting and display threading
//!
//! ## Performance
//!
//! This implementation leverages GPU acceleration through CUDA for high-performance
//! simulation of large Game of Life grids. The convolutional approach allows for
//! efficient parallel computation of neighbor counts and rule application.
//!
//! ## Usage Examples
//!
//! ```bash
//! # Run default simulation (100 iterations)
//! cargo run
//!
//! # Run indefinitely with custom pattern
//! cargo run -- -i glider.txt -n
//!
//! # Run 1000 iterations with custom alive symbol
//! cargo run -- -n 1000 -a '*' -i pattern.txt
//! ```

mod args;
mod inference;
mod model;

use args::Args;
use burn::backend::cuda;
use std::thread;

use clap::Parser;
use inference::convert_to_string;
use model::GolModel;

/// Main entry point for the Game of Life neural network simulation.
///
/// This function orchestrates the entire simulation pipeline:
///
/// 1. **Argument Parsing**: Parses command line arguments for input file, iteration count, and symbols
/// 2. **Device Setup**: Initializes CUDA device for GPU acceleration
/// 3. **Model Creation**: Creates the convolutional neural network model with Game of Life rules
/// 4. **Field Loading**: Loads the initial field state from the specified input file
/// 5. **Output Threading**: Spawns a background thread for real-time output display
/// 6. **Simulation Loop**: Runs the simulation for the specified number of iterations (or indefinitely)
///
/// ## Performance Characteristics
///
/// The simulation uses an optimized approach where:
/// - GPU computation handles the expensive convolution operations
/// - CPU handles string conversion and I/O operations in parallel
/// - Memory allocation is minimized through tensor reuse
///
/// ## Error Handling
///
/// The function will panic if:
/// - CUDA device 0 is not available
/// - The input file cannot be read or is malformed
/// - The output thread receiver disconnects unexpectedly
///
/// ## Threading Model
///
/// The application uses a producer-consumer pattern:
/// - Main thread: Produces new field states via neural network inference
/// - Background thread: Consumes field states and outputs them to stdout
/// - Communication via `std::sync::mpsc::channel` for lock-free message passing
fn main() {
    let args = Args::parse();

    type Backend = cuda::Cuda;
    let device = cuda::CudaDevice::new(0);
    let model: GolModel<Backend> = model::GolModel::default();
    let mut field = args::import_field(&args.input_file, args.alive_symbol, &device).unwrap();

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    thread::spawn(move || inference::output(rx));
    tx.send(convert_to_string(field.clone().bool())).unwrap();
    match args.iterations {
        Some(n) => {
            for _ in 0..n {
                let field_bool = model.forward(field);
                tx.send(convert_to_string(field_bool.clone())).unwrap();
                field = field_bool.float();
            }
        }
        None => loop {
            let field_bool = model.forward(field);
            tx.send(convert_to_string(field_bool.clone())).unwrap();
            field = field_bool.float();
        },
    }
}
