use std::{thread, time::Duration};
use std::sync::mpsc;
use std::time::Instant;

extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;
use show_image::{event, ImageView, ImageInfo, create_window};

mod ann;
use crate::ann::*;

mod layer;
use layer::grid_layer::Layer;

#[show_image::main]
fn main(){   
    let mut layer = Layer::new(100, 100);
    let hood_filter = DMatrix::from_row_slice(3, 3, &[
        1,0,0,
        0,1,0,
        0,0,1,        
    ].map(|x| x as f32));

    println!("{}", hood_filter.row_iter().len()/2);
    // let flatten_fn:  fn(&DMatrix<f32>) -> f32 = 
    fn conway_fn(m: &DMatrix<f32>) -> f32 {        
        let total: f32 = m.as_slice().iter().sum();        
        let alive = m.as_slice()[4] == 1.;
        if alive && (total <= 2. && total >= 1.)  {
            return  1.;
        }else if !alive && total >= 2.{
            return 1.
        }else{
            return 0.5;
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
        let x = (((6.*x).sin()/2.)+0.5) + (rand::thread_rng().gen::<f32>()-1.)/5.;
        return x
    }

    let window = create_window("image", Default::default()).unwrap();
    let debug_w = create_window("debug", Default::default()).unwrap();
    
    loop {
        let m = layer.hood_op(hood_filter.clone(), quadratic_fn);
        
        let layer_image_vec = layer.get_layer_image();
        let image = ImageView::new(ImageInfo::mono8(100, 100), &layer_image_vec);
        
        window.set_image("image", &image);
        println!("{}", m.1.len());
        debug_w.set_image("debug", ImageView::new(
                ImageInfo::mono8(102, 102), 
                &m.1.data.as_vec().clone().iter().map(|x| return (x * 255.) as u8).collect::<Vec<u8>>())
            );
        layer.set_data(m.0);
        thread::sleep(Duration::new(1, 0));
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

fn main1() {
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
    evolve(ann.clone(),1000,100,fitness_test);
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    /*
      
    */

    // println!("{:?}", ann.nodes);
    // let out = ann.propagate(vec![1.,1.]);
    // println!("{:?}", out);
    // println!("{:?}", ann.nodes);

}

fn evolve(ann: ANN, total_gen: i32, gen_pop: u32, fitness_fn: fn(&mut ANN) -> f32){
    let mut generation: Vec<(ANN, f32)> = vec![];
    for _i in 0..gen_pop{

        let mut ann = ann.clone();
        ann.mutate();
        generation.push((ann, -1.));
    }

    for _i in 0..total_gen{
        let pop = gen_pop as usize / 2;
        
        let mut handles = vec![];
        
        for i in 0..pop {
            generation[i].0 = generation[i+ pop].0.clone();
            generation[i].0.mutate();            
        }

        let (tx, rx) = mpsc::channel();
        let mut index = 0;
        for (ann, _fitness) in generation.iter_mut() {            

            let tx_clone = tx.clone();
            let mut ann_clone = ann.clone();

            handles.push(thread::spawn(move || {
                if tx_clone.send((index, fitness_fn(&mut ann_clone))).is_err(){
                    println!("FITNESS TEST FAILED")
                };
            }));
            index+=1;
        }
        drop(tx);
        
        for (index, fitness) in rx {
            //println!("Fitness: {}", fitness);
            generation[index].1 = fitness;            
        }
        for handle in handles {
            if handle.join().is_err(){
                println!("ERROR JOINING THREAD");
            }
        }
        

        generation.sort_by(|x,y| x.1.partial_cmp(&y.1).unwrap());
        //println!("Best of generation has score of: {}", generation[generation.len()-1].1);
    }
    println!("Best of generation has score of: {}", generation[generation.len()-1].1);
}

fn evolve_seq(ann: ANN, total_gen: i32, gen_pop: u32, fitness_fn: fn(&mut ANN) -> f32){
    let mut generation: Vec<(ANN, f32)> = vec![];
    for _i in 0..gen_pop{

        let mut ann = ann.clone();
        ann.mutate();
        generation.push((ann, -1.));
    }

    for _i in 0..total_gen{
        let pop = gen_pop as usize / 2;
        
        for i in 0..pop {
            generation[i].0 = generation[i+ pop].0.clone();
            generation[i].0.mutate();            
        }

        for (ann, fitness) in generation.iter_mut() {
            *fitness = fitness_fn(ann);
        }       

        generation.sort_by(|x,y| x.1.partial_cmp(&y.1).unwrap());
        println!("Best of generation has score of: {}", generation[generation.len()-1].1);
    }
}


fn fitness_test(ann: &mut ANN) -> f32{
    let mut score = 0.;

    score += 0.-ann.propagate(vec![1.,1.])[0];
    score += 0.+ann.propagate(vec![-1.,1.])[0];
    score += 0.+ann.propagate(vec![1.,-1.])[0];
    score += 0.-ann.propagate(vec![-1.,-1.])[0];

    return score;    
}