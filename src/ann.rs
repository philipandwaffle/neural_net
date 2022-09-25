use na::DMatrix;
extern crate nalgebra as na;
use rand::Rng;
use std::fmt::Debug;


//TODO: add getters to properly display ANN's properties, left public for testing
pub struct ANN{
    pub source: Vec<(u32, f32)>,
    //source contains tuples
    //  0th being the source node's id
    //  1st being the edges weight 
    pub index: Vec<[u32; 2]>,
    //index contains arrays of length 2
    //  0th being the dest node's id
    //  1st being the start index of the source array
    pub nodes: Vec<[f32; 2]>,
    //nodes contains arrays of lenght 2
    //  index being the node's id
    //  oth being the nodes bias
    //  1st being the nodes current value
    pub input: Vec<u32>,
    pub output: Vec<u32>,
    pub activation_fn: fn(f32)-> f32,
    pub weight_mutate_fn: fn(f32) -> f32,
    pub bias_mutate_fn: fn(f32) -> f32,
}

impl ANN{
    pub fn new(
        adjacency_matrix: &DMatrix<bool>, 
        input: Vec<u32>, 
        output: Vec<u32>, 
        activation_fn: fn(f32) -> f32,
        weight_mutate_fn: fn(f32) -> f32,
        bias_mutate_fn: fn(f32) -> f32,
        
    ) -> Self{
        let mut coords: Vec<[u32; 2]> = vec![];
        let mut source: Vec<(u32, f32)> = vec![];
        let mut index: Vec<[u32; 2]> = vec![];
        let mut nodes: Vec<[f32; 2]> = vec![];

        let mut dest_id = 0;
        for col in adjacency_matrix.column_iter(){
            nodes.push([rand::thread_rng().gen(), 0.]);
            let mut source_id = 0;
            for edge in col.iter(){
                if edge == &true{
                    coords.push([source_id, dest_id]);
                    source.push((source_id, rand::thread_rng().gen()));
                    if index.is_empty() || index[index.len()-1][0] != dest_id{
                        index.push([dest_id, source.len() as u32 -1]);
                    }
                }
                source_id += 1;
            }
            dest_id += 1;
        }
        println!("{:?}", coords);
        println!("{:?}", source);
        println!("{:?}", index);
        println!("{:?}", nodes);

        return Self{
            source,
            index,
            nodes,
            input,
            output,
            activation_fn,
            weight_mutate_fn,
            bias_mutate_fn,
        };
    }

    pub fn propagate(&mut self, input: Vec<f32>) -> Vec<f32>{
        let input_len = input.len();
        if self.input.len() != input_len{
            panic!("input length mismatch, is {}, should be {}", input_len, self.input.len());
        }

        //setting values of the input nodes
        for i in 0..input_len{
            self.nodes[i][1] = input[i];
        }

        let index_len = self.index.len();
        for i in 0..index_len{
            let dest_id = self.index[i][0];
            let start_i = self.index[i][1];
            let end_i = if index_len < i+1 {
                self.index[i+1][1]
            } else {
                self.source.len() as u32
            };

            self.nodes[dest_id as usize][1] = self.sigma_dest_node(dest_id, start_i, end_i)
        }
        // to do return the output

        let bar = self.output.iter().map(|node_id| self.nodes[*node_id as usize][1]).collect::<Vec<f32>>();
        return bar;
    }

    fn sigma_dest_node(&self, dest_id: u32, start_i: u32, end_i: u32) -> f32{
        let mut total: f32 = 0.;

        for i in start_i as usize..end_i as usize{
            let value = self.nodes[self.source[i].0 as usize][1];
            let weight = self.source[i].1;
            total += value * weight;
        }

        return (self.activation_fn)(total + self.nodes[dest_id as usize][0]);
    }
    
    pub fn mutate(&mut self){
        for weight in self.source.iter_mut(){
            weight.1 = (self.weight_mutate_fn)(weight.1);
        }
        for bias in self.nodes.iter_mut(){
            bias[0] = (self.bias_mutate_fn)(bias[0]);
        }
    }
}
impl Clone for ANN{
    fn clone(&self) -> Self {
        Self { 
            source: self.source.clone(), 
            index: self.index.clone(), 
            nodes: self.nodes.clone(), 
            input: self.input.clone(), 
            output: self.output.clone(), 
            activation_fn: self.activation_fn.clone(),
            weight_mutate_fn: self.weight_mutate_fn.clone(),
            bias_mutate_fn: self.bias_mutate_fn.clone() 
        }
    }
}
impl Debug for ANN{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ANN")
            .field("source", &self.source)
            .field("index", &self.index)
            .field("nodes", &self.nodes)
            .field("input", &self.input)
            .field("output", &self.output)
            .field("activation_fn", &self.activation_fn)
            .field("weight_mutate_fn", &self.weight_mutate_fn)
            .field("bias_mutate_fn", &self.bias_mutate_fn)
        .finish()
    }
}