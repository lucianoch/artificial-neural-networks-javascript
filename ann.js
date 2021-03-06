
function repeat(x, n){
	out = [];
	for(var i = 0; i < n; i++)
		out.push(x);
	return out;
}

function GAMMA(){ return 0.00001; } // weight decay parameter
function ETA(){ return 0.5; } // learning rate
function ITER(){ return 500; } // number of iterations
function U(){ return 0.5; } // momentum
function show_error(){ return true; } // show error on every step of training


function new_ann(n1, n2, n3){
	var ann = {
		n1: n1, // neurons in input layer
		n2: n2, // neurons in hidden layer
		n3: n3, // neurons in output layer
		weights_1: [], // weights of input-layer with hidden-layer; weights_1[a][b]: input-layer[b] -> hidden-layer[a]; [n2][n1]
		weights_2: [], // weights of hidden-layer with output-layer; weights_2[a][b]: hidden-layer[b] -> output-layer[a]; [n3][n2]
		layer_1: repeat(0, n1+1), // input layer; +1 <- bias
		layer_2: repeat(0, n2+1), // hidden layer; +1 <- bias
		layer_3: repeat(0, n3+1), // output layer; +1 <- bias
		show: function(){
			return [
				"n1: " + this.n1 + "; n2: " + this.n2 + "; n3: " + this.n3 + ";",
				"layer_1:" + JSON.stringify(this.layer_1),
				"layer_2:" + JSON.stringify(this.layer_2),
				"layer_3:" + JSON.stringify(this.layer_3),
				"weights_1:" + JSON.stringify(this.weights_1),
				"weights_2:" + JSON.stringify(this.weights_2)
			].join("\n");
		}
	};
	// fill in the weights with random numbers
	for(var i = 0; i < n2+1; i++){
		ann.weights_1[i] = [];
		for(var j = 0; j < n1+1; j++)
			ann.weights_1[i][j] = Math.random()*2 - 1;
	}
	for(var i = 0; i < n3+1; i++){
		ann.weights_2[i] = [];
		for(var j = 0; j < n2+1; j++)
			ann.weights_2[i][j] = Math.random()*2 - 1;
	}
	return ann;
}

function parse_data(text, line_separator, column_separator){
	return text.split(line_separator).map( function(x){ return x.split(column_separator).map(parseFloat); } );
}

function sigmoid(h){ return 1/(1 + Math.exp(-h)); }

function run_ann(ann){
	// hidden layer
	ann.layer_2[0] = -1;
	for(var j = 1; j < ann.n2+1; j++){
		h = 0;
		for(var k = 0; k < ann.n1+1; k++)
			h += ann.weights_1[j][k] * ann.layer_1[k];
		ann.layer_2[j] = sigmoid(h);
	}
	// output layer
	for(var i = 1; i < ann.n3+1;i++){
		h = 0;
		for(var j = 0; j < ann.n2+1;j++)
			h += ann.weights_2[i][j] * ann.layer_2[j];
		ann.layer_3[i] = h ;  // linear activation
	}
}

function propagate(ann, input){

	var mse = 0;
	var input_length = input.length;
	var weight_decay_aux = 0;
	var predictions = []

	for(var inp = 0; inp < input_length; inp++){
		// load the input
		ann.layer_1[0] = -1;
		for(var i = 1; i <= ann.n1; i++) // DIFFERENT INDEX (-1)
			ann.layer_1[i] = input[i-1];
		// run the net
		run_ann(ann);
		// copy the results
		for(var i = 1; i <= ann.n3; i++)
			predictions[i-1] = ann.layer_3[i]; // DIFFERENT INDEX (-1)
	}

	return predictions;

}

function train(ann, xs, ys){ // xs and ys must have the same length (xs: list of inputs, ys: list of targets)

	var grad2 = [];
	var grad3 = [];
	var diff_weight_1 = [];
	var diff_weight_2 = [];

	var xs_length = xs.length;

	for(var j = 1; j <= ann.n2; j++){
		grad2[j] = 0;
		diff_weight_1[j] = [];
		for(var k = 0; k <= ann.n1; k++)
			diff_weight_1[j][k] = 0;
	}
	for(var i = 1; i <= ann.n3; i ++){
		grad3[i] = 0;
		diff_weight_2[i] = [];
		for(var j = 0; j <= ann.n2; j++)
			diff_weight_2[i][j] = 0;
	}
	ann.layer_1[0] = -1; // bias layer 1
	ann.layer_2[0] = -1; // bias layer 2


	var eta = ETA(); // it must change

	var target = [];

	// main loop
	for(var iter = 0; iter < ITER(); iter++){

		// inner loop
		for(var nu = 0; nu < xs_length; nu++){

			// load input
			for(var k = 1; k <= ann.n1; k++)
				ann.layer_1[k] = xs[nu][k-1];

			// run the net
			//console.log(ann.show());
			run_ann(ann);

			// load target
			for(var k = 1; k <= ann.n3; k++)
				target[k] = ys[nu][k-1]; // DIFFERENT INDEX (-1)

			// gradient calculation (layer 3)
			for(var i = 1; i <= ann.n3; i++)
				grad3[i] = (target[i] - ann.layer_3[i]); // linear

			// gradient calculation (layer 2)
			for(var j = 1; j <= ann.n2; j++){
				sum = 0;
				for(var i = 1; i <= ann.n3; i++)
					sum += grad3[i] * ann.weights_2[i][j];
				grad2[j] = sum * (1 - ann.layer_2[j]) * ann.layer_2[j]; // sigmoid
			}

			// diff 2 calculation, adjust weights 2
			for(var i = 1; i <= ann.n3; i++)
				for(var j = 0; j <= ann.n2; j++){
					diff_weight_2[i][j] = U() * diff_weight_2[i][j] + eta * grad3[i] * ann.layer_2[j];
					ann.weights_2[i][j] *= (1 - eta*GAMMA()*2);
					ann.weights_2[i][j] += diff_weight_2[i][j];
				}

			// diff 1 calculation, adjust weights 1
			for(var j = 1; j <= ann.n2; j++)
				for(var k = 0; k <= ann.n1; k++){
					diff_weight_1[j][k] = U() * diff_weight_1[j][k] + eta * grad2[j] * ann.layer_1[k];
					ann.weights_1[j][k] *= (1 - eta*GAMMA()*2);
					ann.weights_1[j][k] += diff_weight_1[j][k];
				}

		} // ends inner loop
		
		if( show_error() ){
			var error = 0;
			for(var nu = 0; nu < xs_length; nu++){

				// load input
				for(var k = 1; k <= ann.n1; k++)
					ann.layer_1[k] = xs[nu][k-1];

				// run the net
				run_ann(ann);

				// load target
				for(var k = 1; k <= ann.n3; k++)
					target[k] = ys[nu][k-1];

				// gradient calculation (layer 3)
				for(var i = 1; i <= ann.n3; i++)
					error += Math.pow(target[i] - ann.layer_3[i], 2);

				console.log(error);

			}
		}

	} // ends main loop

	return 0;

}


// test the algorithm, function: f(x) = x + 1 (binary!)

var the_net = new_ann(3, 6, 3);

function nat2bin(n){ // from 0 to 7
	if(n%2 == 0) p0 = 0; else p0 = 1;
	n = Math.floor(n/2);
	if(n%2 == 0) p1 = 0; else p1 = 1;
	n = Math.floor(n/2);
	if(n%2 == 0) p2 = 0; else p2 = 1;
	return [p2, p1, p0];
}

function bin2nat(xs){ // from 0 to 7
	if(xs[0] < 0.5) xs[0] = 0; else xs[0] = 1;
	if(xs[1] < 0.5) xs[1] = 0; else xs[1] = 1;
	if(xs[2] < 0.5) xs[2] = 0; else xs[2] = 1;
	return xs[0] * 4 + xs[1] * 2 + xs[2] * 1;
}

train_list = [0, 3, 4, 6];

xs = train_list.map(nat2bin);
ys = train_list.map( function(x){ return nat2bin(x + 1); } );

train(the_net, xs, ys);

console.log("Trained (difference, ideal: 1):");
for(var i = 0; i < train_list.length; i++)
	console.log( [bin2nat(propagate(the_net, nat2bin(train_list[i]))) - train_list[i]] );

console.log("All (difference, ideal: 1):");
for(var i = 0; i < 8; i++)
	console.log( [bin2nat(propagate(the_net, nat2bin(i))) - i] );
