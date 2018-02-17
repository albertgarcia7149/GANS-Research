import os
import os.path
import shutil
import tensorflow as tf

LOGDIR = "/tmp/mnist_tutorial/"
LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)
### Get a sprite and labels file for the embedding projector ###

#make sure necessaryu files are present
if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
  print("Necessary data files were not found. Run this command from inside the "
    "repo provided at "
    "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")
  exit(1)

#define a simple convolutional layer
def conv_layer(input, size_in, size_out, name="conv"):
    #apply name scope so all items are grouped in tensorboard
    with tf.name_scope(name):
        #store the weights
        # takes an intial value (tf.truncated_normal([5,5,size_in,size_out]))
        #  tf.truncated_normal takes a tensor for the shape of output tensor ([5,5,size_in,size_out])
        #  tf.truncated_normal takes a stddev of the normal distribution before truncation (stddev=0.1)
        # takes an optional name for the variable ("W")
        w=tf.Variable(tf.truncated_normal([5,5,size_in,size_out], stddev=0.1),name="W")
        #store the biases
        # takes an intial value (tf.constant(0.1,shape=[size_out]),name="B")
        #  tf.constant takes a constant value of output type (0.1)
        #  tf.constant takes a shape for dimensions of resulting tensor (shape=[size_out])
        # takes an optional name for the variable ("B")
        b=tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")
        #apply convolution operation
        # takes an input in form of a tensor (input)
        # takes a filter in a 4d tensor shape needs: [filter_height, filter_width, in_channels, out_channels]  (w)
        # takes a 1d tensor of length 4 [the stride of the sliding window for each dimension of input] (strides=[1,1,1,1])
        # takes a string either [SAME or VALID] to determine padding algorithm  ([padding="SAME"])
        conv=tf.nn.conv2d(input,w,strides=[1,1,1,1], padding="SAME")
        #apply relu to convolution and biases
        # takes a tensor, of a specific type, see documentation (conv+b)
        # takes an optional name for the operation (none)
        act=tf.nn.relu(conv+b)
        #adds a histogram summary in tensorboard
        # takes a name which acts as the node name in tensorboard (weights)
        # takes a real numeric tensor which is used to build histogram (w)
        tf.summary.histogram("weights",w)
        # takes a name which acts as the node name in tensorboard (biases)
        # takes a real numeric tensor which is used to build histogram (b)
        tf.summary.histogram("biases",b)
        # takes a name which acts as the node name in tensorboard (activations)
        # takes a real numeric tensor which is used to build histogram (act)
        tf.summary.histogram("activations",act)
        #Performs the max pooling
        # takes a value, 4D tensor
        # takes a 1D tensor of 4 elements which is the size of the window for each dimension of input tensor (ksize=[1,2,2,1])
        # takes a 1D tensor of 4 elements which is the stride of the sliding window for each dimension of the input tensor (strides=[1,2,2,1])
        # takes a string either [SAME or VALID] to determine padding algorithm  ([padding="SAME"])
        return tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# a fully connected layer
def fc_layer(input,size_in, size_out, name="fc"):
    #apply name scope so all items are grouped in tensorboard
    with tf.name_scope(name):
        #store the weights
        # takes an initial value (tf.truncated_normal([size_in, size_out], stddev=0.1))
        #  tf.truncated_normal takes a tensor for the shape of output tensor ([size_in, size_out], stddev=0.1)
        #  tf.truncated_normal takes a stddev of the normal distribution before truncation (stddev=0.1)
        # takes an optioal name ("W")
        w=tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1),name="W")
        #stores the biases
        # takes initial value (tf.constants(0.1[size_out]))
        #  tf.constant takes a constant value of output type (0.1)
        #  tf.constant takes a shape for dimensions of resulting tensor (shape=[size_out])
        # takes an optional name ("B")
        b=tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        #apply relu to matrix multiplication plus biases
        # takes a tensor, of specific type, see documentation (tf.matmul(input,w)+b)
        #  matmul takes 2 tensors of same rank, both rank 2 in this case, (input,w)
        #  matmul takes optional name (none)
        # takes optional name for operation (none)
        act=tf.nn.relu(tf.matmul(input,w)+b)
        #adds a histogram summary in tensorboard
        # takes a name which acts as the node name in tensorboard (weights)
        # takes a real numeric tensor which is used to build histogram (w)
        tf.summary.histogram("weights",w)
        # takes a name which acts as the node name in tensorboard (biases)
        # takes a real numeric tensor which is used to build histogram (b)
        tf.summary.histogram("biases",b)
        # takes a name which acts as the node name in tensorboard (activations)
        # takes a real numeric tensor which is used to build histogram (act)
        tf.summary.histogram("activations",act)
        return act

#Feed-forward Setup
def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
    #clears the default graph stack and resets the global default graph
    tf.reset_default_graph()
    #define tensorflow session
    sess=tf.Session()
    #setup placeholders, and reshape the data

    #placeholder for input image
    # takes type of elements in the tensor that is going to be fed to the placeholder (tf.float32)
    # takes the shape of the tensor to be fed (shape=[None,784])
    #  None means that, in this case the first dimension or batch size, can be any size (None,784)
    #  784 is the size of the flattened 28x28 image from mnist
    # takes an optional name for the operation (none)
    x=tf.placeholder(tf.float32, shape=[None,784], name="x")
    #reshape image so it is 28x28 and we can convolve on it
    # takes a tensor (x)
    # takes a shape that defines the shape of the output tensor, note for images we can use batch*height*width*color
    #  -1 is a special value that makes the total size remain constant in this case it is used to infer the shape for this example it will allow the batch size to be whatever fits the rest of the parameters
    # takes an optional name (none)
    x_image=tf.reshape(x, [-1,28,28,1])
    #makes a summary image in tensorboard
    # takes a name for new node (input)
    # takes a 4D tensor of shape batch_size x height x width x channels where channels is 1,3 or 4 (x_image)
    # takes a max number of batch elements to generate images for
    tf.summary.image('input',x_image,3)
    #placeholder for input labels
    # takes type of elements in the tensor that is going to be fed to the placeholder (tf.float32)
    # takes the shape of the tensor to be fed (shape=[None,10])
    #  None means that, in this case the first dimension or batch size, can be any size (None,10)
    #  10 is the amount of possible labels, ex 0-9 in this case
    # takes an optional name for the operation
    y=tf.placeholder(tf.float32, shape=[None,10])

    #Create the network
    if use_two_conv:
        #calls the conv_layer function
        conv1=conv_layer(x_image,1 ,32,"conv1")
        #calls the conv_layer function
        conv_out=conv_layer(conv1,32,64,"conv2")
    else:
        #calls the conv_layer function
        conv_out=conv_layer(x_image,1,16,"conv")

    #flattens the second pool layer
    # takes a tensor (pool2)
    # takes a shape that will define the shape of the output tensor ([-1,7*7*64])
    # takes optional name
    flattened=tf.reshape(conv_out, [-1,7*7*64])

    if use_two_fc:
        #define first fully connected layer using fc_layer()
        fc1=fc_layer(flattened, 7*7*64, 1024)
        #define relu funtion for fc1
        relu=tf.nn.relu(fc1)
        #set embedding_input to result of relu function
        embedding_input=relu
        # takes a name which acts as the node name in tensorboard (fc1/relu)
        # takes a real numeric tensor which is used to build histogram (relu)
        tf.summary.histogram("fc1/relu",relu)
        #define second fully connected layer using fc_layer()
        embedding_size=1024
        #define second fully connected layer using fc_layer()
        logits=fc_layer(fc1, 1024, 10, "fc2")
    else:
        #make the embeding input flattened
        embedding_input=flattened
        #set the embeding size
        embedding_size=7*7*64
        #define first fully connected layer
        logits=fc_layer(flattened, 1024, 10, "fc2")

    #apply name scope so all items are grouped in tensorboard
    with tf.name_scope("xent"):
        #Compute cross entrophy as our loss function
        # takes a tensor to reduce (tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        #  softmax takes logits which are unscaled log probabilities (logits=logits)
        #  softmax takes labels (labels=y)
        #  softmax computes the softmax cross entrophy between the 2 inputs (logits=logits, labels=y)
        # reduce mean tries to reduce the result from softmax
        xent=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        #Outputs a summary containing a single scalar value
        # takes a name for the generated node for its name in tensorboard ("xent")
        # takes a tensor containing a single value (xent)
        tf.summary.scalar("xent",xent)

    #apply name scope so all items are grouped in tensorboard
    with tf.name_scope("train"):
        #Use an AdamOptimizer to train the network
        # takes a learning rate (1e-4)
        #  minimize applies the gradient (cross_entropy)
        train_step=tf.train.AdamOptimizer(learning_rate).minimize(xent)

    #apply name scope so all items are grouped in tensorboard
    with tf.name_scope("accuracy"):
        #compute accuracy
        # takes 2 tensors each of the same type and returns a tensor of type bool
        #  argmax takes a tensor as input (logits) (y)
        #  argmax takes a tensor as the axis so it knows what axis to reduce across (1) (1)
        #  argmax takes an optional name (none)
        correct_prediction=tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
        # takes a tensor to reduce (tf.cast(correct_prediction, tf.float32))
        #  tf.cast takes a tensor or a sparse tensor (correct_prediction)
        #  tf.cast takes a destination type of which to cast to (tf.float32)
        # reduce mean tries to reduce the result from tf.cast
        accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #Outputs a summary containing a single scalar value
        # takes a name for the generated node for its name in tensorboard ("accuracy")
        # takes a tensor containing a single value (accuracy)
        tf.summary.scalar("accuracy",accuracy)

    #merge all summaries
    summ=tf.summary.merge_all()
    #set the embedding
    # takes a tensor (tf.zeros([1024, embedding_size]), name="test_embedding")
    #  tf.zero takes a shape for output tensor ([1024, embedding_size])
    #  tf.zero makes a tensor of above shape with all values set to 0
    # takes a name (name="test_embedding")
    embedding=tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    #sets values from one tensor to another
    # takes a (variable tensor).assign (embedding)
    # takes a tensor with values (embedding_input)
    # assigns values from tensor to the variable
    assignment=embedding.assign(embedding_input)
    #create a saver object for saving variables
    saver=tf.train.Saver()

    #Initialize all the variables
    # run a session
    #  tf.global_variables_initializer() intializes global variables in the graph
    sess.run(tf.global_variables_initializer())
    #creates events in a given directory and add summaries and events to it
    # takes a string of a directory where event file will be writen
    writer=tf.summary.FileWriter(LOGDIR + hparam)
    #add a graph to event file
    writer.add_graph(sess.graph)

    #Setup tensorboard embedding visualizations
    config=tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config=config.embeddings.add()
    embedding_config.tensor_name=embedding.name
    embedding_config.sprite.image_path=SPRITES
    embedding_config.metadata_path=LABELS
    #Specify the width and height of a single thumbnail
    embedding_config.sprite.single_image_dim.extend([28,28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    #Train for 2000 steps
    for i in range(2001):
        #gets the next 100 images to process
        batch=mnist.train.next_batch(100)
        if i % 5 ==0:
            #get summary information from the session
            [train_accuracy, s] = sess.run([accuracy,sum], feed_dict={x: batch[0], y: batch[1]})
            #add a summary to the writer
            writer.add_summary(s,i)
        #Occarionally report accuracy (every 500 iterations)
        if i%500==0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

#create the hyperparameter strings
def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    #determine the number of convolutional layers
    conv_param="conv=2" if use_two_conv else "conv=1"
    #determine the number of fully connected layers
    fc_param="fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
    #try for all these learning rates
    for learning_rate in [1E-3, 1E-4]:
        #try for 2 fully connected layers
        for use_two_fc in [True]:
            #try for 2 convolutional layers and 1 convolutional layer
            for use_two_conv in [False, True]:
                #get hyperparamater string
                hparam=make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Start run for %s' % hparam)

                #run with new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)
    print('Done Training!')
    print("Run 'tensorboard --logdir=%s' to see the results." % LOGDIR)

#Start Program
main()

