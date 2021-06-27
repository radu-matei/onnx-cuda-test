use image::Pixel;
use ndarray::{s, Array, ArrayBase};
use std::io::{BufRead, BufReader, Cursor};
use tract_onnx::prelude::*;

pub const SQUEEZENET_PATH: &str = "testdata/models/squeezenet1.1-7.onnx";
pub const MOBILENETV2_PATH: &str = "testdata/models/mobilenetv2-7.onnx";
pub const LABELS_PATH: &str = "testdata/models/squeezenet_labels.txt";
pub const IMG_DIR: &str = "testdata/images/";
pub const IMG_PATH: &str = "testdata/images/n04350905.jpg";

fn main() {
    // run(
    //     std::fs::read(MOBILENETV2_PATH).unwrap(),
    //     IMG_PATH.to_string(),
    // )
    // .unwrap();
    batch();
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
        run(model.clone(), path.to_string_lossy().to_string()).unwrap();
    }
}

fn run(model_bytes: Vec<u8>, image: String) -> TractResult<()> {
    let mut model_bytes = Cursor::new(model_bytes);
    let model = tract_onnx::onnx()
        // load the model
        .model_for_read(&mut model_bytes)?
        // specify input type and shape
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)),
        )?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    println!("trying to load image {:#?}", image);
    let image = image::imageops::resize(
        &image::open(image)?,
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

    let output_tensor = model.run(tvec!(array.into()))?;
    let output_tensor = output_tensor.get(0).unwrap();
    let vec: Vec<f32> = output_tensor.as_slice().unwrap().to_vec();
    let output_tensor = Array::from_shape_vec((1, 1000, 1, 1), vec).unwrap();
    let mut probabilities: Vec<(usize, f32)> = output_tensor
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

    Ok(())
}

pub trait NdArrayTensor<S, T, D> {
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
        <S as ndarray::RawData>::Elem: std::clone::Clone,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign;
}

impl<S, T, D> NdArrayTensor<S, T, D> for ArrayBase<S, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
    <S as ndarray::RawData>::Elem: std::clone::Clone,
    T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
{
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D> {
        let mut new_array: Array<T, D> = self.to_owned();
        new_array.map_inplace(|v| *v = v.exp());
        let sum = new_array.sum_axis(axis).insert_axis(axis);
        new_array /= &sum;

        new_array
    }
}
