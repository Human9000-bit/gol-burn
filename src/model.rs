const DEFAULT_3X3: [[f64; 3]; 3] = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];

use burn::{
    module::{Module, Param},
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{Bool, Float, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct GolModel<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> GolModel<B> {
    /// Create a new game of life model with the given kernel/
    pub fn new(kernel: Tensor<B, 4, Float>, device: &B::Device) -> Self {
        let mut conv = Conv2dConfig::new([1, 1], [3, 3])
            .with_bias(false)
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        conv.weight = Param::from_tensor(kernel);
        Self { conv }
    }

    /// Process a field using the game of life rules.
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
    /// Create a default game of life model with the default kernel.
    /// 
    /// Default kernel:
    /// 
    /// ```
    /// [
    ///     [1.0, 1.0, 1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [1.0, 1.0, 1.0],
    /// ]
    /// ```
    fn default() -> Self {
        let device = &B::Device::default();
        let matrix: Tensor<B, 2, Float> = Tensor::from_floats(DEFAULT_3X3, device);
        let kernel = matrix.reshape([1, 1, 3, 3]);
        Self::new(kernel, device)
    }
}
