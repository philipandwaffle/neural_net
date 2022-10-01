pub mod grid_layer{
    extern crate nalgebra as na;
    use std::{any::TypeId, ops::Mul};

    use na::{DMatrix, OMatrix, Dynamic, ToTypenum, Matrix, DimMax, Const, SliceStorage, RowDVector, DVector};
    use rand::Rng;

    pub struct Layer{
        layer_data: DMatrix<f32>,
    }

    impl Layer{
        pub fn new(nrows: usize, ncols: usize) -> Self{
            return Self{
                layer_data: DMatrix::from_fn(nrows, ncols, |_x,_y| {return rand::thread_rng().gen::<f32>()*1.}),
            };
        }

        pub fn get_point(&self, row: usize, col: usize) -> f32{
            return self.layer_data.row(row)[col];
        }
        pub fn set_data(&mut self, data: DMatrix<f32>){
            self.layer_data = data;
        }

        //neighborhood operation that return the layer after the operation has been applied
        pub fn hood_op(&self, hood_filter: DMatrix<f32>, flatten_fn: fn(&DMatrix<f32>) -> f32) -> (DMatrix<f32>, DMatrix<f32>){
            let rows = self.layer_data.row_iter().len();
            let cols = self.layer_data.column_iter().len();

            let hood_rows = hood_filter.row_iter().len();
            let hood_cols = hood_filter.column_iter().len();
            
            let mut data: Vec<f32> = vec![];

            let looped_rows = rows+(hood_rows/2);
            let looped_cols = cols+(hood_cols/2);

            let mut looped_matrix = self.layer_data.clone();

            //looped_matrix = looped_matrix.resize(looped_rows, looped_cols, 0.);
            let hrr = hood_rows/2;
            let hcr = hood_cols/2;
            println!("{}, {}", hrr, hcr);
            for row in 0..hrr{
                looped_matrix = looped_matrix.insert_row(row, 1.);
                let row_copy: Vec<f32> = looped_matrix.row(looped_matrix.column_iter().count() - row).iter().map(|x| *x).collect();
                looped_matrix.set_row(row, &RowDVector::from_row_slice(&row_copy));

                looped_matrix = looped_matrix.insert_row((looped_rows - row), 1.);
                let row_copy: Vec<f32> = looped_matrix.row(row + hrr).iter().map(|x| *x).collect();
                looped_matrix.set_row((looped_rows - row), &RowDVector::from_row_slice(&row_copy));
            }
            for col in 0..hcr{                
                looped_matrix = looped_matrix.insert_column(col, 1.);                
                let col_copy: Vec<f32> = looped_matrix.column(looped_matrix.column_iter().count() - (col + 1)).iter().map(|x| *x).collect();
                looped_matrix.set_column(col, &DVector::from_column_slice(&col_copy));

                looped_matrix = looped_matrix.insert_column((looped_cols - col), 1.);
                let col_copy: Vec<f32> = looped_matrix.column(col + hcr).iter().map(|x| *x).collect();
                looped_matrix.set_column((looped_cols - col), &DVector::from_column_slice(&col_copy));
            }            
            println!("hello");
            for row in 0..rows{
                for col in 0..cols{
                    let hood: Matrix<f32, Dynamic, Dynamic, na::SliceStorage<f32, Dynamic, Dynamic, na::Const<1>, Dynamic>>;
                    hood = looped_matrix.slice((row,col), (hood_rows, hood_cols));

                    let mut hood_result = DMatrix::repeat(hood_cols, hood_rows, 0.);                    
                    hood_filter.mul_to(&hood, &mut hood_result);

                    let value = (flatten_fn)(&hood_result);
                    data.push(value);
                }                
            }

            return (DMatrix::from_row_slice(rows, cols, &data), looped_matrix);
        }

        pub fn get_layer_image(&self) -> Vec<u8>{            
            return self.layer_data.data.as_vec().clone().iter().map(|x| return (x * 255.) as u8).collect::<Vec<u8>>();
        }
    }
}