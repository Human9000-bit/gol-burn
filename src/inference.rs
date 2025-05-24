use std::io::Write;

use burn::{prelude::Backend, tensor::{Bool, Tensor}};

const SQUARE_SYMBOL: char = '■';

pub fn output(rx: std::sync::mpsc::Receiver<String>) {
    loop {
        let message = rx.recv().unwrap();
        std::io::stdout().write_all(message.as_bytes()).unwrap();
    }
}

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