mod args;
mod inference;
mod model;

use args::Args;
use burn::backend::cuda;
use std::thread;

use clap::Parser;
use inference::convert_to_string;
use model::GolModel;

fn main() {
    let args = Args::parse();

    type Backend = cuda::Cuda;
    let device = cuda::CudaDevice::new(0);
    let model: GolModel<Backend> = model::GolModel::default();
    let mut field = args::import_field(&args.input_file, args.alive_symbol, &device).unwrap();

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    thread::spawn(move || inference::output(rx));
    for _ in 0..args.iterations {
        let field_bool = model.forward(field);
        tx.send(convert_to_string(field_bool.clone())).unwrap();
        field = field_bool.float();
    }
}
