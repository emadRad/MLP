### Multi layer Perzeptron
#### Requirement
* JDK(Java Development Kit) 1.8

#### How to Run
To run this program in terminal first compile it with javac:
```
$ javac MLP.java
```

then run it with two arguments:
```
$ java MLP [Data File] 
```

for instance:
```
$ java MLP Train/training2.dat 
```

#### Input Files Format
You need to provide two input files, one for the input data and the other for weights.

The input data for computation is as follows:

3 3 1

1 0 1

0 0 1

1 0 1


The first line contains the number of data(P),the input dimension(N) and the output dimension(M), in this example P=3, N=3, M=1.

The input data X comes after the first line and each line contains an instance of the input X.  For this input we have 3 instance of X.


** Note: You can follow the the provided pattern for input file(e.g. training2.dat) but add number of data and dimension of input and output to
the first line after the comments(lines started with #).

### Multi Layer Perczeptron Configuration

In the MLP.java file in the main function you can change the array net_config in order to change the configuration of network.The number of neuron 
at each layer are stored in this array(input array in index 0).

You can also change the configuration of each layer by changing the instance of configuration class (or creating a new instance).
For example the following code shows a configuration of a layer with transfer function tanh and learning rate of 0.3.

```java
	Configuration h1_config=new Configuration(Layer.Function.TANH,0.3);
```


Author:Emad Bahrami Rad

Email: emadbahramirad@gmail.com
