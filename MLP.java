import java.math.MathContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import java.io.*;



class MLP{
	ArrayList<double []> X = new ArrayList<>();
	ArrayList<double []> Y_Teacher = new ArrayList<>();
	ArrayList<double []> out_all;

	ArrayList<Layer> network;


	double [] errors;
	
	//N = X dimension
	int N;
	
	//M = Y dimension
	int M;	
	
	//P = number of input instance
	int P;

	
	public MLP(InputReader in){
		ReadInput(in);		
	}


	public void Initialize(int [] config,ArrayList<Configuration> layers_config){
		Layer layer;
		network = new ArrayList<>();
		int h,m;
		Configuration layer_conf;
		
		//N without BIAS
		config[0] = N-1;
		config[config.length-1]= M;

		for(int i=1;i<config.length;i++){
			h=config[i-1]+1;
			m=config[i];
			layer_conf = layers_config.get(i-1);
			layer = new Layer(h,m,layer_conf.getFunction(),layer_conf.getEta());
			network.add(layer);
		}
	}


	/* Computing net_k 
	 *  w_nk*out_n, sum over n
	 *
	 *  weight matrix n*k
	 *  out_n 1*n
	 *
	 */ 
	public double [] ForwardPropagate(double [] out_n,Layer layer_k){
		double [][] w_nk = layer_k.getWeightMatrix();
		double [] out_k =new double[w_nk[0].length+1]; 
		Arrays.fill(out_k,0);

		//index 0 is the bias value,setting BIAS value 
		out_k[0]=1.0;

		for(int k=0;k<out_k.length-1;k++)
			for(int n=0;n<out_n.length;n++){
				out_k[k+1] += out_n[n]*w_nk[n][k];	
		}

		return out_k;
	}

	// @input: y_teacher for pattern p
	public void BackwardPropagte(double [] y_teacher){
		
		//ignore index 0 in output layer
		Layer out_Layer= network.get(network.size()-1);	 	
		

		double [] df = out_Layer.computeDerivative();	
		

		for(int j=0;j<y_teacher.length;j++)
		{
			out_Layer.setDelta((y_teacher[j]-out_Layer.getOut()[j+1])*df[j],j);
		}
		
		Layer layer_below = out_Layer;
		Layer current_layer;
		int L = network.size()-1;
		double weightPenalty=0;
		// index 0 is the layer after input layer
		for(int l=L-1;l>=0;l--)
		{
			current_layer = network.get(l);
			df = current_layer.computeDerivative();
			for(int h=0;h<current_layer.net.length;h++)
			{
				for(int k=0;k<layer_below.delta.length;k++)
				{
					weightPenalty+=layer_below.delta[k]*layer_below.weights[h][k];
				}
				current_layer.setDelta(weightPenalty*df[h],h);
			}

		}

	}


	public void updateWeights(){

		Layer layer;
		double [] out_i;
		for(int l=0;l<network.size();l++){
			layer=network.get(l);
			for(int i=0;i<layer.weights.length;i++){
				out_i = out_all.get(l);	
				for(int j=0;j<layer.weights[0].length;j++){
					layer.weights[i][j]+=out_i[i]*layer.getEta()*layer.delta[j];
				}
			}
		}	
	
	}


	public void RunAlgorithm(){
		Layer layer_k;
		double error_per_pattern=0;
		double [] y_teacher;
		double [] out;
		int i;
		out_all = new ArrayList<>();
		//looping through each input pattern p stored in X 
		for(int p=0;p<X.size();p++)
		{
			out= X.get(p);
			y_teacher = Y_Teacher.get(p);
			for(i=0;i<network.size();i++){
				out_all.add(out);
				layer_k = network.get(i);
				
				//System.out.println("Layer "+ (i+1)+" neurons: "+layer_k.num_of_neurons);
				
				layer_k.setNet(ForwardPropagate(out,layer_k));
				layer_k.eval_trans_func();
				out = layer_k.getOut();
				
				//for(int j=0;j<out.length;j++)
				//	System.out.print(out[j]+" ");
				//System.out.println();
				


			
			}	
			
			errors[p]=computeError(y_teacher,out);
		
			BackwardPropagte(y_teacher);
			
			updateWeights();		
			
		}

	}


	public double computeError(double [] y_teacher, double [] y)
	{
		double e=0;
		/* out_Layer.out array contains the output of layer out_Layer
		*  the index 0 of the array 'out' for each layer is the BIAS value(=1.0)
		*  thus, the ouput of network(y) is started from index 1 
		*/ 
		for(int j=1;j<y.length;j++){
			e += Math.pow(y_teacher[j-1]-y[j],2);
		}
		e = 0.5*e;

		return e;
	
	}

	public void ReadInput(InputReader in){
		String line;
		String [] tokens=null ;
		


		P = in.nextInt();
		N = in.nextInt();
		M = in.nextInt();
		errors = new double[P];

		double [] XX;
		double [] YY;
		String [] lineTokens;
		
		for(int p=0;p<P;p++)
		{
			XX = new double[N+1];
			
			//adding BIAS value
			XX[0]=1.0;
			for(int n=1;n<N+1;n++){
				XX[n] = in.nextDouble();
			}
			X.add(XX);
			YY = new double[M];	
			for(int m=0;m<M;m++)
				YY[m] = in.nextDouble();
			Y_Teacher.add(YY);
		}	
		

		N++;
		
	}
	

	public void Learn()
	{
		long seed = System.nanoTime();
		int num_of_iter=1;
		double []error_per_iter=new double[num_of_iter];	
		double global_error=0;
		
		for(int i=0;i<num_of_iter;i++)
		{
			RunAlgorithm();
			for(int p=0;p<P;p++)
				global_error+=errors[p];
			global_error=global_error/P;
			
			error_per_iter[i]=global_error;	

			// Shuffling both X and Y_Teacher(training pattern) that keeps
			// the matching between X and Y_Teacher
			Collections.shuffle(X,new Random(seed));
			Collections.shuffle(Y_Teacher,new Random(seed));
		}
		
		File curve=new File("learning.curve");
		String str="";
		try( FileWriter fileWriter = new FileWriter(curve,false)){
			/*	
			for(int i=0;i<num_of_iter;i++){
				str=(i+1)+" "+error_per_iter[i]+"\n";
				fileWriter.write(str);
			}
			*/
			
			
			for(int p=0;p<P;p++){
				str=(p+1)+" "+errors[p]+"\n";
				fileWriter.write(str);
			}
			
		
		}
		catch(IOException e){
			System.out.println("Error in writing to file"+e);
		}
	
	}

	public double Evaluate(){
		Layer layer_k;
		double [] y_teacher=null;
		double [] out=null;
		int i;
		out_all = new ArrayList<>();
		for(int p=0;p<X.size();p++)
		{
			out= X.get(p);
			y_teacher = Y_Teacher.get(p);
			for(i=0;i<network.size();i++){
				out_all.add(out);
				layer_k = network.get(i);
				layer_k.setNet(ForwardPropagate(out,layer_k));
				layer_k.eval_trans_func();
				out = layer_k.getOut();
			}
		}	
		return computeError(y_teacher,out);
	}


	public static void main(String [] args)throws FileNotFoundException
	{
		
		if(args.length==0 || args[0].isEmpty())
				throw new FileNotFoundException("File not found! Please check the file name and the path to input file(or number of inputs)");
	
		InputStream inputStream;
	
		try {
			inputStream = new FileInputStream(args[0]);
	
		}
		catch (IOException e) {
		 throw new RuntimeException(e);
		}
		
		InputReader in = new InputReader(inputStream,"#");

		// index 0 : input layer and so forth
		
		//TODO : network with one hidden layer	
		int [] net_config = {2,4,1};		
		//int [] net_config = {4,4,4,2};

		MLP mlp = new MLP(in);
		Configuration h1_config=new Configuration(Layer.Function.TANH,0.3);
		//Configuration h2_config=new Configuration(Layer.Function.LOGISTIC,0.2);
		Configuration output_config=new Configuration(Layer.Function.TANH,0.1);
		ArrayList<Configuration> layers =new ArrayList<>(Arrays.asList(h1_config,output_config));
		
		System.out.println("Configuration of the network: ");
		System.out.println("Numer of input: "+net_config[0]);
		System.out.println("Number of neuron in the first hidden layer "+net_config[1]);
		System.out.println("Number of neuron in the second hidden layer "+net_config[2]);
		
		//System.out.println("Numer of ouputs: "+net_config[3]);


		mlp.Initialize(net_config,layers);

		Scanner scanner = new Scanner(System.in);
		String input;
		boolean correct=false;
		do{
			System.out.println("Do you want to train a dataset or evaluate?\n(Enter T for train or E to evaluate, accroding to the input files or Q to exit)");
			input = scanner.next();
			if(input.toUpperCase().trim().equals("T")){
				mlp.Learn();
				System.out.println("The data wrote to learning.curve");	
				//correct = true;
				
			}
			if(input.toUpperCase().trim().equals("E")){
				System.out.println("Enter the path to the input file name for evaluation:\n(please consider the number of input and output) ");
				input=scanner.next();
				
				try {
					inputStream = new FileInputStream(input);
	
				}
				catch (IOException e) {
		 			throw new RuntimeException(e);
				}
				in = new InputReader(inputStream,"#");
				mlp.ReadInput(in);	
				System.out.println("The evaluation error:"+mlp.Evaluate());	
				//correct = true;
			}
			if(input.toUpperCase().trim().equals("Q"))
				correct=true;
			if(!correct)
				System.out.println("Please enter C or T according to the input file you provided:");
		}while(!correct);
	
	}

}


class Configuration{
	private Layer.Function transfer_function;
	private double eta;

	public Configuration(Layer.Function func, double eta){
		this.transfer_function=func;
		this.eta=eta;
	}
	public double getEta(){ return eta;}
	public Layer.Function getFunction(){return transfer_function;}
}





class InputReader{
StringTokenizer tokenizer;
BufferedReader reader;
String skipLineChar;

public InputReader(InputStream stream,String skipChar){
	reader = new BufferedReader(new InputStreamReader(stream));
	tokenizer = null;
	skipLineChar = skipChar;
}
public String readLine(){
		try{
			return reader.readLine();		
		}
		catch (IOException error){
			throw new RuntimeException(error);
		}
	}


public Iterator<String> FileIterator(){
		try{
			return reader.lines().iterator();		
		}
		catch (Exception error){
			throw new RuntimeException(error);
		}
	}

public String next(){
	String line;
	while(tokenizer == null || !tokenizer.hasMoreTokens()) {
		try{
			line= reader.readLine();
			while(line.startsWith(skipLineChar))
				line=reader.readLine();
			tokenizer = new StringTokenizer(line);
		}
		catch (IOException error){
			throw new RuntimeException(error);
		}
	
	}
	return tokenizer.nextToken();
}


public int nextInt(){
	return Integer.parseInt(next());
}
public double nextDouble(){
	return Double.parseDouble(next());
}
}
