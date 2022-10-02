use std::{thread, time::Duration};
use std::sync::mpsc;
use std::time::Instant;

extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;
use show_image::{event, ImageView, ImageInfo, create_window};

mod ai;

mod layer;
use layer::automata::Layer;

use crate::ai::ai::{ANN, evolve, evolve_seq};

static layer_x: usize = 500;
static layer_y: usize = 500;

//#[show_image::main]
fn main1(){    
    let hood_filter = DMatrix::from_row_slice(5, 5, &[
        1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,    
        0,0,0,1,0,    
        0,0,0,0,1,    
    ].map(|x| x as f32));
    let hood_filter = DMatrix::from_row_slice(3, 3, &[
        1,0,0,
        0,1,0,
        0,0,1,
    ].map(|x| x as f32));

    fn point_gen_fn(x: usize, y: usize) -> f32 {
        if x%50 == 0 || y%20 == 0{
            return 1.;
        }else{
            return 0.
        }
    }
    fn rand_gen_fn(x: usize, y: usize) -> f32 {
        if rand::thread_rng().gen::<f32>() >= 0.2{
            return 1.;
        }else{
            return 0.;
        }
    }
    
    let mut layer = Layer::new(layer_x, layer_y, rand_gen_fn);

    println!("{}", hood_filter.row_iter().len()/2);
    // let flatten_fn:  fn(&DMatrix<f32>) -> f32 = 
    fn conway_fn(m: &DMatrix<f32>) -> f32 {
        let total: f32 = m.as_slice().iter().sum();
        let alive = m.as_slice()[4] == 1.;
        //println!("Total: {}, alive: {}", total, alive);
        if alive && (total == 3. && total == 4.)  {
            return  1.;
        }else if !alive && total == 2.{
            return 1.
        }else{
            return 0.;
        }
    }
    fn avg_fn(m: &DMatrix<f32>) -> f32 {
        let total: f32 = m.as_slice().iter().sum();
        return total/9.;
    }
    fn func(m: &DMatrix<f32>) -> f32 {
        let total: f32 = m.as_slice().iter().sum();
        return (total/9.).tanh();
    }
    fn quadratic_fn(m: &DMatrix<f32>) -> f32 {
        let total: f32 = m.as_slice().iter().sum();
        return 4.*((total/9.)-0.5).powf(2.);
    }
    fn sin_fn(m: &DMatrix<f32>) -> f32 {
        let x: f32 = m.as_slice().iter().sum();
        let x = (x/9.).tanh();
        let x = (((6.*x).sin()/2.)+0.5);// + (rand::thread_rng().gen::<f32>()-1.)/5.;
        return x
    }

    let window = create_window("image", Default::default()).unwrap();
    let debug_w = create_window("debug", Default::default()).unwrap();
    
    loop {
        let m = layer.hood_op(hood_filter.clone(), conway_fn);
        
        let layer_image_vec = layer.get_layer_image();
        let image = ImageView::new(ImageInfo::mono8(layer_x as u32, layer_y as u32), &layer_image_vec);
        
        window.set_image("image", &image);
        println!("{}", m.1.len());
        debug_w.set_image("debug", ImageView::new(
                ImageInfo::mono8(layer_x as u32 + 2, layer_y as u32 + 2), 
                &m.1.data.as_vec().clone().iter().map(|x| return (x * 255.) as u8).collect::<Vec<u8>>())
            );
        layer.set_data(m.0);
        //thread::sleep(Duration::new(0, ));
    }

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            println!("{:#?}", event);
            if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
                break;
            }
        }
    }      

    

    
}

fn main() {
    println!("Hello, world!");

    let adj_m = DMatrix::from_row_slice(13, 13, &[
        0,0,1,1,1,1,1,1,1,1,1,0,0,        
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,1,1,1,1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,
    ].map(|x| x != 0));
    //let activation_fn: fn(f32) -> f32 = |x| {return 1. / (1. + 2.718281828459045f32.powf(x))};
    let activation_fn: fn(f32) -> f32 = |x| {x.tanh()};
    let weight_mutate_fn: fn(f32) -> f32 = |x| {return x + rand::thread_rng().gen_range(-0.05..=0.05)};
    let bias_mutate_fn: fn(f32) -> f32 = |x| {return x + rand::thread_rng().gen_range(-0.05..=0.05)};
        
    let mut ann = ANN::new(
        &adj_m, 
        vec![0,1], 
        vec![12], 
        activation_fn,
        weight_mutate_fn,
        bias_mutate_fn,
    );
    ann.propagate(vec![-1.,-1.])[0];
    let start = Instant::now();
    evolve(ann.clone(),1000,100,fitness_test);
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);

    let start = Instant::now();
    evolve_seq(ann.clone(),1000,100,fitness_test);
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
}




fn fitness_test(ann: &mut ANN) -> f32{
    let mut score = 0.;

    score += 0.-ann.propagate(vec![1.,1.])[0];
    score += 0.+ann.propagate(vec![-1.,1.])[0];
    score += 0.+ann.propagate(vec![1.,-1.])[0];
    score += 0.-ann.propagate(vec![-1.,-1.])[0];

    return score;    
}