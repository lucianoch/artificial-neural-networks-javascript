
function new_ann(n1, n2, n3){
	var ann = {
		n1: n1, // neurons in input layer
		n2: n2, // neurons in hidden layer
		n3: n3, // neurons in output layer
		weights1: [], // weights of input-layer with hidden-layer
		weights2: [], // weights of hidden-layer with output-layer
		show: function(){
			return [
				"n1: " + this.n1 + "; n2: " + this.n2 + "; n3: " + this.n3 + ";",
				JSON.stringify(this.weights1),
				JSON.stringify(this.weights2)
			].join("\n");
		}
	};
	// fill in the weights with random numbers
	for(var i = 0; i < n1; i++){
		ann.weights1[i] = [];
		for(var j = 0; j < n2; j++){
			ann.weights1[i][j] = Math.random()*2 - 1;
		}
	}
	for(var i = 0; i < n2; i++){
		ann.weights2[i] = [];
		for(var j = 0; j < n3; j++){
			ann.weights2[i][j] = Math.random()*2 - 1;
		}
	}
	return ann;
}
/*
		int ITER: 500,           // Total de Iteraciones
		float ETA: 0.5,          // learning rate
		float u: 0.5,            // Momentum
		float GAMMA: 0.5,        // weight-decay
*/


