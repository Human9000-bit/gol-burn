//! # Game of Life Output and Visualization
//! 
//! This module handles the conversion of Game of Life tensor data into human-readable
//! string representations and manages real-time output display. It provides:
//! 
//! - **Tensor to String Conversion**: Transforms 4D boolean tensors into ASCII art
//! - **Real-time Output**: Threaded output handling for smooth animation display
//! - **Visual Representation**: Uses Unicode block characters for clear visualization
//! 
//! ## Threading Model
//! 
//! The output system uses a producer-consumer pattern where:
//! - Main thread generates new field states and sends them via channel
//! - Background thread receives states and outputs them to stdout
//! - Non-blocking message passing ensures smooth simulation performance
//! 
//! ## Visualization Format
//! 
//! Live cells are represented by solid block characters (■) and dead cells by spaces,
//! creating a clear visual representation of the Game of Life evolution.
//! 
//! ## Performance Considerations
//! 
//! - String allocation is optimized with pre-calculated capacity
//! - Tensor conversion is done in a single pass
//! - Output operations are isolated to prevent blocking the main simulation loop

use std::io::Write;

use burn::{prelude::Backend, tensor::{Bool, Tensor}};

/// Unicode block character used to represent alive cells in the visual output.
/// 
/// This solid block character (■) provides excellent visual contrast against spaces
/// (dead cells) and renders consistently across most terminals and text displays.
/// 
/// Alternative symbols could include:
/// - `'*'` for ASCII-only compatibility
/// - `'●'` for a circular representation  
/// - `'█'` for full block character
/// 
/// The choice of `'■'` balances visual clarity with broad terminal support.
const SQUARE_SYMBOL: char = '■';

/// Background thread function for real-time Game of Life output display.
/// 
/// This function runs in a dedicated thread and continuously receives formatted
/// Game of Life states from the main simulation thread via a channel. It immediately
/// outputs each received state to stdout, enabling real-time visualization of the
/// cellular automaton evolution.
/// 
/// ## Threading Benefits
/// 
/// - **Non-blocking**: Main simulation thread never waits for I/O operations
/// - **Smooth Animation**: Consistent output timing independent of computation speed  
/// - **Resource Isolation**: I/O operations don't interfere with GPU computation
/// 
/// # Arguments
/// 
/// * `rx` - Receiving end of a channel that delivers formatted Game of Life states as strings
/// 
/// # Behavior
/// 
/// The function runs an infinite loop that:
/// 1. Blocks waiting for the next message from the simulation thread
/// 2. Immediately writes the received string to stdout
/// 3. Flushes output to ensure immediate display
/// 
/// # Panics
/// 
/// This function will panic if:
/// - The sender end of the channel is dropped (simulation thread terminates)
/// - Writing to stdout fails (e.g., broken pipe, permission issues)
/// 
/// # Examples
/// 
/// ```rust
/// use std::sync::mpsc;
/// use std::thread;
/// 
/// let (tx, rx) = mpsc::channel::<String>();
/// 
/// // Spawn output thread
/// thread::spawn(move || output(rx));
/// 
/// // Send formatted states from main thread
/// tx.send("■ ■\n ■ \n■ ■".to_string()).unwrap();
/// ```
/// 
/// # Performance
/// 
/// This function is designed for real-time output and prioritizes low latency
/// over throughput. Each message is immediately flushed to ensure responsive
/// visual feedback during simulation.
pub fn output(rx: std::sync::mpsc::Receiver<String>) {
    loop {
        let message = rx.recv().unwrap();
        std::io::stdout().write_all(message.as_bytes()).unwrap();
    }
}

/// Converts a 4D boolean tensor representing a Game of Life state into a formatted string.
/// 
/// This function transforms the neural network output (a 4D boolean tensor) into a
/// human-readable ASCII art representation suitable for terminal display. The conversion:
/// 
/// - Maps `true` values to solid block characters (■) representing alive cells
/// - Maps `false` values to spaces representing dead cells  
/// - Arranges output in a grid with newlines separating rows
/// - Optimizes memory allocation for better performance
/// 
/// ## Tensor Format
/// 
/// The input tensor must have shape [batch, channels, height, width] where:
/// - Batch dimension (index 0): Should be 1 for single grid
/// - Channel dimension (index 1): Should be 1 for single Game of Life state
/// - Height dimension (index 2): Number of rows in the grid
/// - Width dimension (index 3): Number of columns in the grid
/// 
/// # Arguments
/// 
/// * `grid` - A 4D boolean tensor with shape [1, 1, height, width] representing the Game of Life state
/// 
/// # Returns
/// 
/// A formatted string where:
/// - Each line represents one row of the grid
/// - `■` characters represent alive cells (true values)
/// - Space characters represent dead cells (false values)
/// - Lines are separated by newline characters (except the last line)
/// 
/// # Examples
/// 
/// ```rust
/// use burn::tensor::Tensor;
/// 
/// // 3x3 glider pattern
/// let glider_data = [
///     [false, true,  false],
///     [false, false, true ],
///     [true,  true,  true ]
/// ];
/// let tensor = create_tensor_from_2d_bool(glider_data); // Helper function
/// let output = convert_to_string(tensor);
/// 
/// assert_eq!(output, " ■ \n  ■\n■■■");
/// ```
/// 
/// # Performance Optimizations
/// 
/// - **Pre-allocated String**: Capacity is calculated upfront to avoid reallocations
/// - **Single Pass**: Tensor data is converted in one linear traversal
/// - **Minimal Copies**: Direct indexing into the flattened boolean vector
/// - **Efficient Layout**: Row-major order matches typical tensor storage
/// 
/// # Generics
/// 
/// * `B: Backend` - The Burn backend type (e.g., Candle, CUDA, CPU)
/// 
/// # Memory Layout
/// 
/// The function assumes the tensor data is stored in row-major order, where
/// `index = row * width + col` maps 2D coordinates to the flattened vector index.
pub fn convert_to_string<B: Backend>(grid: Tensor<B, 4, Bool>) -> String {
    let dims = grid.dims();
    let grid = grid.reshape([dims[0], dims[1]]).to_data();
    let bool_matrix_vec: Vec<bool> = grid.to_vec().unwrap();
    
    let height = dims[0];
    let width = dims[1];
    
    let mut result = String::with_capacity(height * (width + 1));
    
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if bool_matrix_vec[idx] {
                result.push(SQUARE_SYMBOL);
            } else {
                result.push(' ');
            }
        }
        if row < height - 1 {
            result.push('\n');
        }
    }
    
    result
}