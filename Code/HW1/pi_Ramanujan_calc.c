#include <stdio.h>
#include <math.h>

int fact(int n) {
  int result = 1;
  for (int i = 1; i <= n; i++) {
    result *= i;
  }
  return result;
}

int main() {
  int n = 100;
  double pii = 0.0;
  for (int k = 0; k <= n; k++) {
    pii += (double)fact(4*k)*(1103+26390*k)/(fact(k)*fact(4*k)*pow(396, 4*k));
  }
  pii *= 2.0/9801;

  printf("Pi = %lf\n", pii);
  
  return 0;
}