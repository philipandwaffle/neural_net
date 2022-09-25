use na::DMatrix;
extern crate nalgebra as na;

fn main() {
    println!("Hello, world!");
    
    let adj_m = DMatrix::from_row_slice(7, 7, &[
        0,0,1,1,1,0,0,        
        0,0,1,1,1,0,0,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
    ].map(|x| x != 0));
    let sigmoid: fn(f32) -> f32 = |x| return 1. / 1. + 2.718281828459045f32.powf(x);

    //NeuralNetCSR::new(&adj_m, vec![0,1], vec![5,6]);
    
    NeuralNetCSC::new(&adj_m, vec![0,1], vec![5,6], sigmoid);
}


struct NeuralNetCSR{
    index: Vec<[u32; 2]>,
    //index contains arrays of length 2
    //  0th being the source node's id
    //  1st being the start index of the dest array
    dest: Vec<(u32, f32)>,
    //dest contains tuples
    //  0th being the destination node's id
    //  1st being the edges weight 
    nodes: Vec<[f32; 2]>,
    //nodes contains arrays of lenght 2
    //  index being the node's id
    //  oth being the nodes bias
    //  1st being the nodes current value
    input: Vec<u32>,
    output: Vec<u32>,
}

impl NeuralNetCSR{
    pub fn new(adjacency_matrix: &DMatrix<bool>, input: Vec<u32>, output: Vec<u32>) -> Self{
        let mut coords: Vec<[u32; 2]> = vec![];
        let mut dest: Vec<(u32, f32)> = vec![];
        let mut index: Vec<[u32; 2]> = vec![];
        let mut nodes: Vec<[f32; 2]> = vec![];

        let mut source_id = 0;
        for row in adjacency_matrix.row_iter(){
            nodes.push([1., 0.]);
            let mut dest_id = 0;
            for edge in row.iter(){
                if edge == &true{
                    coords.push([source_id, dest_id]);
                    dest.push((dest_id, 1.));
                    if index.is_empty() || index[index.len()-1][0] != source_id{
                        index.push([source_id, dest.len() as u32 -1]);
                    }
                }
                dest_id += 1;
            }
            source_id += 1;
        }
        println!("{:?}", coords);
        println!("{:?}", index);
        println!("{:?}", dest);
        println!("{:?}", nodes);

        return Self{
            index,
            dest,
            nodes,
            input,
            output,
        };
    }

    fn propagate(input: Vec<f32>) -> Vec<f32>{
        todo!();
    }
    
}

struct NeuralNetCSC{
    source: Vec<(u32, f32)>,
    //source contains tuples
    //  0th being the source node's id
    //  1st being the edges weight 
    index: Vec<[u32; 2]>,
    //index contains arrays of length 2
    //  0th being the dest node's id
    //  1st being the start index of the source array
    nodes: Vec<[f32; 2]>,
    //nodes contains arrays of lenght 2
    //  index being the node's id
    //  oth being the nodes bias
    //  1st being the nodes current value
    input: Vec<u32>,
    output: Vec<u32>,
    activation_func: fn(f32)-> f32,
}

impl NeuralNetCSC{
    pub fn new(adjacency_matrix: &DMatrix<bool>, input: Vec<u32>, output: Vec<u32>, activation_func: fn(f32) -> f32) -> Self{
        let mut coords: Vec<[u32; 2]> = vec![];
        let mut source: Vec<(u32, f32)> = vec![];
        let mut index: Vec<[u32; 2]> = vec![];
        let mut nodes: Vec<[f32; 2]> = vec![];

        let mut dest_id = 0;
        for col in adjacency_matrix.column_iter(){
            nodes.push([1., 0.]);
            let mut source_id = 0;
            for edge in col.iter(){
                if edge == &true{
                    coords.push([source_id, dest_id]);
                    source.push((source_id, 1.));
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
            activation_func,
        };
    }

    pub fn propagate(&mut self, input: Vec<f32>) -> Vec<f32>{
        let input_len = input.len();
        if self.index.len() != input_len{
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
        return self.nodes[];
    }

    fn sigma_dest_node(&self, dest_id: u32, start_i: u32, end_i: u32) -> f32{
        let mut total: f32 = 0.;

        for i in start_i as usize..end_i as usize{
            let value = self.nodes[self.source[i].0 as usize][1];
            let weight = self.source[i].1;
            total += value * weight;
        }

        return (self.activation_func)(total + self.nodes[dest_id as usize][0]);
    }
    
}