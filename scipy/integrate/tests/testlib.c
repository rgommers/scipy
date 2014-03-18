#include "math.h"
const double PI = 3.14159265358979323846264338327950288419716939937510582;
double typical(int n, double args[n]){
  return cos(args[1]*args[0]-args[2] * sin(args[0]))/PI;
}

double indefinite(int n, double args[n]){
	return -exp(-args[0])*log(args[0]);
}

double sin2(int n, double args[n]){
	return sin(args[0]);
}

int main(){
  return 0;
}
