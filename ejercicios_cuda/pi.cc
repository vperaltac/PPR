#include <iostream>
#include <iomanip>

static long num_steps = 100000;
double step;

int main(){
    double x,sum = 0.0;
    double pi;

    step = 1.0/(double) num_steps;
    for(int i=1; i<= num_steps; i++){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;

    std::cout << std::setprecision(100) << "PI = " << pi << std::endl; 
}