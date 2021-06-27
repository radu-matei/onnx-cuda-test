use image::Pixel;
use ndarray::s;
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use std::io::{BufRead, BufReader};

pub const SQUEEZENET_PATH: &str = "testdata/models/squeezenet1.1-7.onnx";
pub const MOBILENETV2_PATH: &str = "testdata/models/mobilenetv2-7.onnx";
pub const LABELS_PATH: &str = "testdata/models/squeezenet_labels.txt";
pub const IMG_DIR: &str = "testdata/images/";
pub const IMG_PATH: &str = "testdata/images/n04350905.jpg";

fn main() {
    run(
        std::fs::read(MOBILENETV2_PATH).unwrap(),
        IMG_PATH.to_string(),
    );
    // batch();
}

fn batch() {
    let mut entries = std::fs::read_dir(IMG_DIR)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, std::io::Error>>()
        .unwrap();
    entries.sort();
    let model = std::fs::read(MOBILENETV2_PATH).unwrap();

    for path in entries.iter() {
        run(model.clone(), path.to_string_lossy().to_string());
    }
}

fn run(model: Vec<u8>, image: String) {
    let environment = Environment::builder()
        .with_name("integration_test")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .unwrap();

    let mut session = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_memory(model)
        .unwrap();

    println!("trying to load image {:#?}", image);
    let image = image::imageops::resize(
        &image::open(image).unwrap(),
        224,
        224,
        ::image::imageops::FilterType::Triangle,
    );

    println!("resized image: {:#?}", image.dimensions());

    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    // let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
    // let output0_shape: Vec<usize> = session.outputs[0]
    //     .dimensions()
    //     .map(|d| d.unwrap())
    //     .collect();

    // Batch of 1
    let input_tensor_values = vec![array];

    // Perform the inference
    let outputs: Vec<onnxruntime::tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> =
        session.run(input_tensor_values).unwrap();

    let mut probabilities: Vec<(usize, f32)> = outputs[0]
        .softmax(ndarray::Axis(1))
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let labels = BufReader::new(std::fs::File::open(LABELS_PATH).unwrap());

    let labels: Vec<String> = labels.lines().map(|line| line.unwrap()).collect();

    for i in 0..5 {
        println!(
            "class={} ({}); probability={}",
            labels[probabilities[i].0], probabilities[i].0, probabilities[i].1
        );
    }
}
