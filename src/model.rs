//! # Game of Life Neural Network Model
//!
//! This module implements Conway's Game of Life using a convolutional neural network approach.
//! The core insight is that the Game of Life rules can be efficiently computed using:
//!
//! 1. **Convolution**: Count neighbors using a 3x3 kernel
//! 2. **Boolean Logic**: Apply survival and birth rules via tensor operations
//!
//! ## Algorithm
//!
//! The Game of Life follows these rules:
//! - **Survival**: A live cell with 2 or 3 neighbors survives
//! - **Birth**: A dead cell with exactly 3 neighbors becomes alive
//! - **Death**: All other cells die or remain dead
//!
//! ## Neural Network Implementation
//!
//! This implementation uses a single convolutional layer to count neighbors, then applies
//! the Game of Life rules using tensor boolean operations. This approach is highly efficient
//! for large grids and leverages GPU acceleration.
//!
//! ## Performance Benefits
//!
//! - **Parallel Processing**: All cells computed simultaneously via convolution
//! - **GPU Acceleration**: Tensor operations run on CUDA/GPU backends
//! - **Memory Efficient**: In-place operations minimize memory allocation
//! - **Vectorized**: No explicit loops over individual cells

/// Default 3x3 convolution kernel for counting neighbors in Game of Life.
///
/// This kernel counts all 8 neighbors around each cell by setting weights to 1.0
/// for all positions except the center (which is 0.0 to exclude the cell itself).
///
/// Layout:
/// ```text
/// [1.0, 1.0, 1.0]
/// [1.0, 0.0, 1.0]  
/// [1.0, 1.0, 1.0]
/// ```
///
/// When convolved with a binary grid (0.0/1.0), this produces the neighbor count
/// for each cell, which is essential for applying Game of Life rules.
const DEFAULT_3X3: [[f64; 3]; 3] = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];

use burn::{
    module::{Module, Param},
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{Bool, Float, Tensor},
};

/// Neural network model implementing Conway's Game of Life via convolution.
///
/// This model uses a single 2D convolutional layer to efficiently compute the next
/// generation of a Game of Life grid. The convolution counts neighbors for each cell,
/// and then boolean tensor operations apply the survival and birth rules.
///
/// ## Architecture
///
/// - **Input**: 4D tensor [batch, channels, height, width] with float values (0.0 or 1.0)
/// - **Convolution**: 3x3 kernel with padding=1 to count neighbors
/// - **Output**: 4D boolean tensor representing the next generation
///
/// ## Generics
///
/// - `B: Backend`: The Burn backend type (e.g., Candle, CUDA, CPU)
///
/// ## Example
///
/// ```rust
/// let model = GolModel::<Backend>::default();
/// let next_gen = model.forward(current_generation);
/// ```
#[derive(Module, Debug)]
pub struct GolModel<B: Backend> {
    /// 2D convolutional layer for neighbor counting.
    ///
    /// Configured with:
    /// - Input channels: 1 (single Game of Life state)
    /// - Output channels: 1 (neighbor counts)
    /// - Kernel size: 3x3
    /// - Padding: 1 (to preserve grid dimensions)
    /// - Bias: false (not needed for simple counting)
    conv: Conv2d<B>,
}

impl<B: Backend> GolModel<B> {
    /// Creates a new Game of Life model with a custom convolution kernel.
    ///
    /// This constructor allows you to specify a custom 3x3 kernel for neighbor counting,
    /// enabling experimentation with variants of the Game of Life or other cellular automata.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A 4D tensor with shape [1, 1, 3, 3] containing the convolution weights
    /// * `device` - The Burn device where the model should be allocated
    ///
    /// # Returns
    ///
    /// A new `GolModel` instance with the specified kernel weights.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn::tensor::Tensor;
    ///
    /// // Create custom kernel (standard Game of Life)
    /// let kernel_data = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];
    /// let kernel = Tensor::from_floats(kernel_data, &device).reshape([1, 1, 3, 3]);
    /// let model = GolModel::new(kernel, &device);
    /// ```
    ///
    /// # Kernel Format
    ///
    /// The kernel tensor must have shape [1, 1, 3, 3] where:
    /// - First dimension: Output channels (always 1)
    /// - Second dimension: Input channels (always 1)  
    /// - Third/Fourth dimensions: 3x3 spatial kernel
    pub fn new(kernel: Tensor<B, 4, Float>, device: &B::Device) -> Self {
        let mut conv = Conv2dConfig::new([1, 1], [3, 3])
            .with_bias(false)
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        conv.weight = Param::from_tensor(kernel);
        Self { conv }
    }

    /// Computes the next generation of the Game of Life grid.
    ///
    /// This method implements the core Game of Life algorithm using neural network operations:
    ///
    /// 1. **Neighbor Counting**: Convolves the input grid to count neighbors for each cell
    /// 2. **Rule Application**: Applies survival and birth rules using boolean tensor operations
    /// 3. **State Transition**: Returns the next generation as a boolean tensor
    ///
    /// ## Game of Life Rules
    ///
    /// - **Birth**: Dead cell (0.0) with exactly 3 neighbors becomes alive (true)
    /// - **Survival**: Live cell (1.0) with 2 or 3 neighbors stays alive (true)  
    /// - **Death**: All other cells become dead (false)
    ///
    /// # Arguments
    ///
    /// * `grid` - Input grid with shape [batch, channels, height, width] and float values (0.0 or 1.0)
    ///
    /// # Returns
    ///
    /// Boolean tensor with the same shape representing the next generation state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let model = GolModel::<Backend>::default();
    /// let current = load_initial_state(); // [1, 1, H, W] tensor
    /// let next = model.forward(current);  // [1, 1, H, W] boolean tensor
    /// ```
    ///
    /// # Performance
    ///
    /// This operation is highly optimized for parallel execution:
    /// - Convolution runs in parallel across all cells
    /// - Boolean operations are vectorized
    /// - GPU acceleration available on compatible backends
    pub fn forward(&self, grid: Tensor<B, 4, Float>) -> Tensor<B, 4, Bool> {
        // Step 1: Convolve the grid with the kernel to get the number of neighbors
        let neighbors = self.conv.forward(grid.clone());

        // Step 2: Apply the rules of the game of life using tensor operations
        let alive = grid.clone().equal_elem(1.0);

        // Survival: (neighbors == 2 | neighbors == 3) & alive
        let survive = neighbors
            .clone()
            .equal_elem(2.0)
            .bool_or(neighbors.clone().equal_elem(3.0));

        // Birth: neighbors == 3 & dead
        let birth = neighbors.equal_elem(3.0).bool_and(grid.equal_elem(0.0));

        // Combine conditions
        birth.bool_or(survive.bool_and(alive))
    }
}

impl<B: Backend> Default for GolModel<B> {
    /// Creates a default Game of Life model with the standard neighbor-counting kernel.
    ///
    /// This is the most common way to create a `GolModel` instance. It uses the standard
    /// 3x3 kernel that counts all 8 neighbors around each cell, implementing the classic
    /// Conway's Game of Life rules.
    ///
    /// ## Default Kernel
    ///
    /// The kernel counts all surrounding neighbors:
    /// ```text
    /// [1.0, 1.0, 1.0]
    /// [1.0, 0.0, 1.0]
    /// [1.0, 1.0, 1.0]
    /// ```
    ///
    /// ## Device Selection
    ///
    /// Uses the backend's default device. For GPU backends, this typically selects
    /// the first available GPU device.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Create default model
    /// let model = GolModel::<Backend>::default();
    ///
    /// // Use for inference
    /// let next_generation = model.forward(current_grid);
    /// ```
    ///
    /// # Equivalent To
    ///
    /// ```rust
    /// let device = &Backend::Device::default();
    /// let kernel = Tensor::from_floats(DEFAULT_3X3, device).reshape([1, 1, 3, 3]);
    /// let model = GolModel::new(kernel, device);
    /// ```
    fn default() -> Self {
        let device = &B::Device::default();
        let matrix: Tensor<B, 2, Float> = Tensor::from_floats(DEFAULT_3X3, device);
        let kernel = matrix.reshape([1, 1, 3, 3]);
        Self::new(kernel, device)
    }
}
