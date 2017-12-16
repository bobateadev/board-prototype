## Use Tensboard to launch PaddlePaddle's data

This is a prototype to demonstrate the feasibility to use Tensorboard to display PaddlePaddle's training data metrics.

The proposed architecture design doc is here:
https://github.com/PaddlePaddle/board/wiki/Architecture-design-for-using-TensorBoard-with-PaddlePaddle

## Use PaddleFileWriter in Paddle script

In paddle script, initializes ```PaddleFileWriter``` and call ```write``` function to log data. Provide ```name``` to group all data into a single graph and ```step``` to plot the data properly.

In mnist.py, it is logging the cost and error evaluator every 10 batches.


## Generate event file

After you run the script(python mnist.py) with PaddleFileWriter write function, a event file "events.out.tfevents.{timestamp}" will be generated in the same program directory. The event file will first write a event with Version and each event for each time we write a value. Each event will be written as a protocol buffer (interface defined in tensorboard.proto) with a particular format of CRC mask function for checksum.


## Launch TensorBoard

After installing tensorflow, run command tensorboard --logdir={event_file_dir}. Go to browser and goes to localhost:6006 to view the graph. TensorBoard runs CRC checksum to verify data integrity before reading the actual data.  If everything is successful, you should be able to see plotted graph(s) in Scalar tab. You can mouse over, zoom in see the details of the metrics you log. 

TensorBoard looks at the entire file directory and search for sub directories. You can place multiple event files in different sub directories for different projects to compare graphs on the same dashboard. 
