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

    //NeuralNetCSR::new(&adj_m, vec![0,1], vec![5,6]);
    NeuralNetCSC::new(&adj_m, vec![0,1], vec![5,6]);
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
}

impl NeuralNetCSC{
    pub fn new(adjacency_matrix: &DMatrix<bool>, input: Vec<u32>, output: Vec<u32>) -> Self{
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
        };
    }

    fn propagate(input: Vec<f32>) -> Vec<f32>{
        todo!();
    }
    
}