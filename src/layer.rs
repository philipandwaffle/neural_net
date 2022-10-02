pub mod automata{
    extern crate nalgebra as na;    

    use na::{DMatrix, Dynamic, Matrix, RowDVector, DVector, VecStorage};
    use rand::Rng;

    use std::arch::x86_64::_MM_MASK_UNDERFLOW;
    use std::sync::Arc;
    use std::{thread, time::Duration};
    use std::sync::mpsc::{self, Receiver, Sender};

    // layer struct contains a matrix with the information
    pub struct Layer{
        layer_data: DMatrix<f32>,
    }

    impl Layer{
        pub fn new(nrows: usize, ncols: usize, gen_fn: fn(usize, usize) -> f32) -> Self{
            return Self{
                layer_data: DMatrix::from_fn(nrows, ncols, gen_fn),
            };
        }

        // set the current data of the layer's matrix
        pub fn set_data(&mut self, data: DMatrix<f32>){
            self.layer_data = data;
        }

        //neighborhood operation that return the layer after the operation has been applied
        pub fn hood_op(&self, hood_filter: DMatrix<f32>, flatten_fn: fn(&DMatrix<f32>) -> f32) -> (DMatrix<f32>, DMatrix<f32>){
            let rows = self.layer_data.row_iter().len();
            let cols = self.layer_data.column_iter().len();

            let hood_rows = hood_filter.row_iter().len();
            let hood_cols = hood_filter.column_iter().len();

            let looped_matrix = self.gen_looped_matrix(hood_cols, hood_rows);
            let mut handles = vec![];
            let (tx, rx) = mpsc::channel();
            for row in 0..rows{
                let arc_looped_matrix: Arc<Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>> = Arc::new(looped_matrix.clone());

                let tx_clone = tx.clone();
                let hood_filter_clone = hood_filter.clone();
                handles.push( thread::spawn(move || {
                    let mut row_data: Vec<f32> = vec![];
                    for col in 0..cols{
                        let hood: Matrix<f32, Dynamic, Dynamic, na::SliceStorage<f32, Dynamic, Dynamic, na::Const<1>, Dynamic>>;
                        hood = arc_looped_matrix.slice((row,col), (hood_rows, hood_cols));

                        let mut hood_result = DMatrix::repeat(hood_cols, hood_rows, 0.);
                        hood_filter_clone.mul_to(&hood, &mut hood_result);

                        let value = (flatten_fn)(&hood_result);
                        row_data.push(value);
                    }
                    tx_clone.send((row_data, row));
                }));
            }
            drop(tx);

            let mut unsorted_data: Vec<(Vec<f32>, usize)> = vec![];
            for row_data in rx {
                unsorted_data.push(row_data);
            }
            unsorted_data.sort_by(|a,b| a.1.cmp(&b.1));

            let mut data: Vec<f32> = vec![];
            for mut row in unsorted_data{
                data.append(&mut row.0);
            }

            for handle in handles {
                if handle.join().is_err(){
                    println!("ERROR JOINING THREAD");
                }
            }

            return (DMatrix::from_row_slice(rows, cols, &data), looped_matrix);
        }

        fn gen_looped_matrix(&self, hood_rows: usize, hood_cols: usize) -> DMatrix<f32>{
            let mut looped_matrix = self.layer_data.clone();

            // the radius of the neighborhood
            let hrr = hood_rows/2;
            let hcr = hood_cols/2;

            // inserting rows from the end to the start
            for row in 0..hrr{
                looped_matrix = looped_matrix.insert_row(row, 1.);
                let cur_rows = looped_matrix.row_iter().count();
                println!("rows: {}, taking from index: {}, putting into index: {}", cur_rows, (cur_rows - (row + 1)), row);
                let row_copy: Vec<f32> = looped_matrix.row(cur_rows - (row + 1)).iter().map(|x| *x).collect();
                looped_matrix.set_row(row, &RowDVector::from_row_slice(&row_copy));
            }

            // inserting rows from the start to the end
            for row in 0..hrr{
                let cur_rows = looped_matrix.row_iter().count();
                looped_matrix = looped_matrix.insert_row(cur_rows, 1.);
                let cur_rows = looped_matrix.row_iter().count();
                println!("rows: {}, taking from index: {}, putting into index: {}", cur_rows, (row + hrr), (cur_rows -1));
                let row_copy: Vec<f32> = looped_matrix.row(row + hrr).iter().map(|x| *x).collect();
                looped_matrix.set_row(cur_rows - 1, &RowDVector::from_row_slice(&row_copy));
            }
            
            // inserting columns from the end to the start
            for col in 0..hcr{
                looped_matrix = looped_matrix.insert_column(col, 1.);
                let cur_cols = looped_matrix.column_iter().count();
                println!("cols: {}, taking from index: {}, putting into index: {}", cur_cols, (cur_cols - (col + 1)), col);
                let col_copy: Vec<f32> = looped_matrix.column(cur_cols - (col + 1)).iter().map(|x| *x).collect();
                looped_matrix.set_column(col, &DVector::from_row_slice(&col_copy));
            }

            // inserting columns from the start to the end
            for col in 0..hcr{
                let cur_cols = looped_matrix.column_iter().count();
                looped_matrix = looped_matrix.insert_column(cur_cols, 1.);
                let cur_cols = looped_matrix.column_iter().count();
                println!("cols: {}, taking from index: {}, putting into index: {}", cur_cols, (col + hcr), (cur_cols -1));
                let col_copy: Vec<f32> = looped_matrix.column(col + hcr).iter().map(|x| *x).collect();
                looped_matrix.set_column(cur_cols - 1, &DVector::from_row_slice(&col_copy));
            }
            return looped_matrix;
        }

        // gets the layer data in the form of a vector to be displayed
        pub fn get_layer_image(&self) -> Vec<u8>{
            return self.layer_data.data.as_vec().clone().iter().map(|x| return (x * 255.) as u8).collect::<Vec<u8>>();
        }
    }
}