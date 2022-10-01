pub mod grid_layer{
    extern crate nalgebra as na;
    use std::{any::TypeId, ops::Mul};

    use na::{DMatrix, OMatrix, Dynamic, ToTypenum, Matrix};

    pub struct Layer{
        layer_data: DMatrix<f32>,
    }

    impl Layer{
        pub fn new(nrows: usize, ncols: usize) -> Self{
            return Self{
                layer_data: DMatrix::repeat(nrows, ncols, 0.),
            };
        }

        pub fn get_point(&self, row: usize, col: usize) -> f32{
            return self.layer_data.row(row)[col];
        }

        //neighborhood operation that return the layer after the operation has been applied
        pub fn hood_op(&self, matrix: DMatrix<f32>, flatten_fn: fn(&DMatrix<f32>) -> f32) -> DMatrix<f32>{
            let rows = self.layer_data.row_iter().len();
            let cols = self.layer_data.column_iter().len();

            let hood_rows = matrix.row_iter().len();
            let hood_cols = matrix.column_iter().len();
            
            let mut data: Vec<f32> = vec![];

            for row in 0..rows-hood_rows{
                for col in 0..cols-hood_cols{
                    let mut hood: Matrix<f32, Dynamic, Dynamic, na::SliceStorage<f32, Dynamic, Dynamic, na::Const<1>, Dynamic>>;
                    
                    // if row == 0 {
                        
                    // }else if row + hood_rows > rows || col + hood_cols > cols {

                    // }else {
                    // }
                    hood = self.layer_data.slice((row,col), (hood_rows, hood_cols));

                    let mut hood_result = DMatrix::repeat(hood_cols, hood_rows, 0.);
                    println!("Hood slice ({},{}) {:?}",row,col,hood);
                    matrix.mul_to(&hood, &mut hood_result);
                    let value = (flatten_fn)(&hood_result);
                    data.push(value);
                }
                println!("{:?}",data);
            }

            return DMatrix::from_row_slice(rows-hood_rows, cols-hood_cols, &data);
        }
    }
}