import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;


/* 
 * index 0: contains the BIAS value or wieght 
 *
 *
 */ 
class Layer{
	/* n*k, [n][k] weights between layer n and k
	*  The weights for neuron k(at layer K) is stored at column k
	*  n together with k, weights[n][k] specifies the weight
	*  for the input to this neuron(or output of previous neuron)
	*
	*/
	public double [][] weights;
	

	public int num_of_neurons;
	
	public enum Function {LOGISTIC,TANH,IDENTITY};

	private Function transfer_Function;

	public double [] delta;

	private double eta;

	/* computed output of each neuron at this layer
	*  index 0 of the 'out' array is the BIAS value(1.0)
	*/
	public double [] out;


	/* The input to each neuron of this layer(or the output of previous layer)
	 * 
	*/
	public double [] net;

	final double startRange = -2.0;
	final double endRange = 2.0;


	public Layer(int h,int m,Function transfer,double eta){
		num_of_neurons = m;
		weights = new double[h][m];
		delta = new double[m]; 
		InitializeWeights();
		transfer_Function = transfer;
		this.eta=eta;
				
		//PrintWeights();	
	}

	public void PrintWeights(){

		for(int i=0;i<weights.length;i++){
			for(int j=0;j<weights[i].length;j++)
				System.out.print(weights[i][j]+" ");	
		System.out.println();	
		}
		System.out.println();
	
	
	}


	private void InitializeWeights(){
		Random rand = new Random();
		rand.setSeed(84213590);	
		for(int h=0;h<weights.length;h++)
			for(int m=0;m<weights[h].length;m++)
				weights[h][m]=getRandomInRange(startRange,endRange,rand);
	}

	
	private double getRandomInRange(double start,double end,Random rand){
		if(start>end)
			throw new IllegalArgumentException("Start cannot exceed End.");

		double range = end - start;	
		double randNum = start+(range * rand.nextDouble());

		return randNum;
	
	}	

	public double getWeight_nk(int n,int k){
		
		if((n>=0 && n<weights.length) && (k>=0 && k<weights[0].length) )	
			return weights[n][k];
		else
			throw new IllegalArgumentException("getWeight_hk index out of bound"); 
	}

	public double [][] getWeightMatrix(){return weights;}


	/* Computing the transfer function on 'out'(the net) 
	 * and storing the output for the below layer in out
	 *
	 */
	public void eval_trans_func(){
		switch(transfer_Function){
			case TANH:
				computeTanh();
				break;
			case LOGISTIC:
				computeLogistic();
				break;
		}	
	}

	public void computeTanh(){
		for(int i=0;i<net.length;i++){
			out[i+1]=Tanh(net[i]);
		}
	}

	public double Tanh(double z){
		double expX = Math.exp(z);
		double exp_X = Math.exp(-z);	
		return (expX-exp_X)/(expX+exp_X);
	
	}

	public double Logistic(double z){
		return 1/(1+Math.exp(-z));
	}

	public void computeLogistic(){
		for(int i=0;i<net.length;i++)
			out[i+1]=Logistic(net[i]); 
	}


	
	public double[] computeDerivative(){
		switch(transfer_Function){
			case TANH:
				return TanhDerivative();
			case LOGISTIC:
				return LogisticDerivative();
		}	

		double [] df = new double[net.length];
		Arrays.fill(df,1.0);
		return df;

	}

	private double [] TanhDerivative(){

		double [] df=new double[net.length];

		//computing the derivative for each neuron of this layer
		// 1-(tanh(z))^2 ,z = weighted sum for this layer net_m
		for(int j=0;j<net.length;j++)
			df[j]=1-Math.pow(Tanh(net[j]),2);
		
		return df;
	
	}

	private double[] LogisticDerivative(){

		double [] df=new double[net.length];
		for(int j=0;j<net.length;j++)
			df[j]=Logistic(net[j])*(1-Logistic(net[j]));	

		return df;
	}
	


	public void setNet(double [] Net){
		setOut(Net);
		net = new double[Net.length-1];
		System.arraycopy(Net,1,net,0,Net.length-1);
	}


	//computed output of each neuron at this layer
	public void setOut(double [] outPut){
		out = new double[outPut.length];
		System.arraycopy(outPut,0,out,0,outPut.length);
	}

	public double [] getOut(){
		return out;
	}

	public double [] getNet(){
		return net;
	}

	public void setDelta(double d,int index){
		delta[index] = d;
	}


	public double getEta(){return eta;}


}
