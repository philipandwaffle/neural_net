use na::DMatrix;
extern crate nalgebra as na;
use rand::Rng;

mod ann;
use crate::ann::*;


fn main() {
    println!("Hello, world!");

    let adj_m = DMatrix::from_row_slice(6, 6, &[
        0,0,1,1,1,0,        
        0,0,1,1,1,0,
        0,0,0,0,0,1,
        0,0,0,0,0,1,
        0,0,0,0,0,1,
        0,0,0,0,0,0,
    ].map(|x| x != 0));
    let activation_fn: fn(f32) -> f32 = |x| {return 1. / (1. + 2.718281828459045f32.powf(x))};
    let weight_mutate_fn: fn(f32) -> f32 = |x| {return x + rand::thread_rng().gen_range(-0.5..=0.5)};
    let bias_mutate_fn: fn(f32) -> f32 = |x| {return x + rand::thread_rng().gen_range(-0.5..=0.5)};
        
    let mut generation: Vec<(ANN, f32)> = vec![];
    let mut ann = ANN::new(
        &adj_m, 
        vec![0,1], 
        vec![5], activation_fn,
        weight_mutate_fn,
        bias_mutate_fn,
    );

    print!("{:?}", ann);

    println!("{}", ann.propagate(vec![-1.,-1.])[0]);
    for i in 0..100{
        let mut ann = ann.clone();
        ann.mutate();
        generation.push((ann.clone(), -1.));
    }

    for (ann, fitness) in generation.iter_mut() {
        let f = fitness_test(ann);
        *fitness = fitness_test(ann);
        println!("{}", f);
    }


    // println!("{:?}", ann.nodes);
    // let out = ann.propagate(vec![1.,1.]);
    // println!("{:?}", out);
    // println!("{:?}", ann.nodes);

}

fn fitness_test(ann: &mut ANN) -> f32{
    let mut score = -1.;
    score += if ann.propagate(vec![1.,1.])[0] < 0.{
        0.25   
    }else {
        -0.25
    };
    score += if ann.propagate(vec![1.,-1.])[0] >= 0.{
        0.25   
    }else {
        -0.25
    };
    score += if ann.propagate(vec![-1.,1.])[0] >= 0.{
        0.25   
    }else {
        -0.25
    };
    score += if ann.propagate(vec![-1.,-1.])[0] < 0.{
        0.25   
    }else {
        -0.25
    };
    return score;    
}