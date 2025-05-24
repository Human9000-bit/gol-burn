//! # Command Line Arguments and Input File Processing
//!
//! This module provides command line argument parsing and file input functionality for the
//! Game of Life neural network implementation. It handles:
//!
//! - Parsing command line arguments for input file, iterations, and cell symbols
//! - Loading Game of Life grids from text files into Burn tensors
//! - Validating input file format and grid consistency
//!
//! ## Example Usage
//!
//! ```
//! use clap::Parser;
//! use gol_burn::args::Args;
//!
//! let args = Args::parse();
//! let field = args::import_field(&args.input_file, args.alive_symbol, &device)?;
//! ```

use std::{fs, path::PathBuf};

use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
use clap::Parser;

/// Command line arguments for the Game of Life neural network simulation.
///
/// This structure defines all configurable parameters that can be passed via command line
/// when running the Game of Life simulation. Uses `clap` for automatic argument parsing
/// and help generation.
///
/// ## Examples
///
/// ```bash
/// # Run with default settings (100 iterations, input.txt, 'X' for alive cells)
/// cargo run
///
/// # Run indefinitely with custom input file
/// cargo run -- -i my_pattern.txt -n
///
/// # Run 500 iterations with '*' as alive symbol
/// cargo run -- -n 500 -a '*' -i glider.txt
/// ```
#[derive(Parser)]
pub struct Args {
    /// File path to field to import
    ///
    /// The input file should contain a rectangular grid where each character represents
    /// a cell. The file format should use newlines to separate rows and should have
    /// consistent row lengths.
    #[arg(short = 'i', long, default_value = "input.txt")]
    pub input_file: PathBuf,

    /// Number of iterations to run
    ///
    /// If not specified or set to `None`, the simulation will run indefinitely.
    /// When specified, the simulation will run for exactly this many generations.
    #[arg(short = 'n', long, default_value = None)]
    pub iterations: Option<usize>,

    /// Symbol representing alive cells
    ///
    /// Any character in the input file that matches this symbol will be considered
    /// a living cell (value 1.0). All other characters represent dead cells (value 0.0).
    #[arg(short = 'a', long, default_value = "X")]
    pub alive_symbol: char,
}

/// Imports a Game of Life field from a text file and converts it to a 4D Burn tensor.
///
/// This function reads a text file where each character represents a cell in the Game of Life grid.
/// The `alive_symbol` character represents living cells (converted to 1.0), while all other
/// characters represent dead cells (converted to 0.0). Newlines are used as row separators.
///
/// The resulting tensor is formatted for use with convolutional neural networks, with batch
/// and channel dimensions added to support the Burn framework's expectations.
///
/// # Arguments
///
/// * `path` - Path to the input text file containing the Game of Life grid
/// * `alive_symbol` - Character that represents alive cells in the input file
/// * `device` - Burn device where the tensor should be allocated (CPU/CUDA/etc.)
///
/// # Returns
///
/// A 4D tensor with shape [1, 1, height, width] where:
/// - Dimension 0: Batch size (always 1)
/// - Dimension 1: Channels (always 1, representing a single Game of Life state)
/// - Dimension 2: Height (number of rows in the grid)
/// - Dimension 3: Width (number of columns in the grid)
///
/// Values are 1.0 for alive cells and 0.0 for dead cells.
///
/// # Errors
///
/// Returns `std::io::Error` if:
/// - The file cannot be opened or read
/// - The file contains invalid UTF-8 sequences
/// - The grid has inconsistent row lengths (all rows must have the same width)
/// - The file contains no valid grid data (empty or only whitespace)
/// - Any other I/O operation fails
///
/// # Examples
///
/// ```
/// use burn::backend::Candle;
/// use std::path::PathBuf;
///
/// let device = Default::default();
/// let path = PathBuf::from("glider.txt");
/// let field = import_field::<Candle>(&path, 'X', &device)?;
/// assert_eq!(field.dims(), [1, 1, 3, 3]); // 3x3 glider pattern
/// ```
pub fn import_field<B: Backend>(
    path: &PathBuf,
    alive_symbol: char,
    device: &B::Device,
) -> Result<Tensor<B, 4, Float>, std::io::Error> {
    // Read file as UTF-8 string (safer than byte conversion)
    let content = fs::read_to_string(path)?;

    // Split into lines and filter out empty lines
    let lines: Vec<&str> = content.lines().filter(|line| !line.is_empty()).collect();

    if lines.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "File contains no valid grid data",
        ));
    }

    // Validate that all rows have the same length
    let width = lines[0].len();
    let height = lines.len();

    for (i, line) in lines.iter().enumerate() {
        if line.len() != width {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Row {} has length {}, expected {}", i, line.len(), width),
            ));
        }
    }

    // Pre-allocate the data vector for better performance
    let mut grid_data = Vec::with_capacity(height * width);

    // Convert characters to floats in a single pass
    for line in lines {
        for ch in line.chars() {
            grid_data.push(if ch == alive_symbol { 1.0 } else { 0.0 });
        }
    }

    // Create tensor directly from the flattened data and reshape
    let tensor = Tensor::<B, 1>::from_floats(grid_data.as_slice(), device)
        .reshape([height, width])
        .unsqueeze::<3>() // Add batch dimension
        .unsqueeze_dim(1); // Add channel dimension

    Ok(tensor)
}
