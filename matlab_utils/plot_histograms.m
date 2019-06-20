clc;
clear all;
close all;

H = [923, 8532, 249, 80, 51, 30, 22, 20, 15, 5, 7, 3, 5, 1, 0, 0, 0, 0, 0, 0, 1];
x = 0:0.05:1;
H = H./sum(H);

figure;
bar(x, H)
xlabel('Difference Ratio')
ylabel('Percentage of total images')

%%%%% result for all dataset
H2 = [301657, 5598395, 179250,70880,39012,25429,17004,12221,8810,6434,4797,3614,2773,1966,1380,861,523,303,200,159,1393];
x = 0:0.05:1;
H2 = H2./sum(H2);

figure;
bar(x, H2)
xlabel('Difference Ratio')
ylabel('Percentage of total images')

%%%%% impressed-weighted for all dataset

H2= [
2680753850
40655821458
1239385926
489120031
271636524
180678055
120607324
91211776
66033778
44357190
34617416
27021962
19148493
13357027
8973228
5093834
3140394
1507532
1349336
897022
8498286];

x = 0:0.05:1;
H2 = H2./sum(H2);

figure;
bar(x, H2)
xlabel('Difference Ratio')
ylabel('Percentage of total images')


%%%%%%%%%%%%%%%%%%% Merchant histogram %%%%%%%%%%%%%%%5
H1 = [
   5.08250825e-02
   8.09240924e-01
   8.38283828e-02
   2.37623762e-02
   1.25412541e-02
   6.60066007e-03
   3.30033003e-03
   3.96039604e-03
   1.98019802e-03
   6.60066007e-04
   6.60066007e-04
   0.00000000e+00
   6.60066007e-04
   0.00000000e+00
   1.32013201e-03
   0.00000000e+00
   0.00000000e+00
   6.60066007e-04
   0.00000000e+00
   0.00000000e+00
   0.00000000e+00];

x = 0:0.05:1;
H1 = H1./sum(H1);

figure;
bar(x, H1)
xlabel('Difference Ratio')
ylabel('Percentage of merchants')


%%%%%%%%%%%%%%%%%%% impression-weighted Merchant histogram %%%%%%%%%%%%%%%5
H2 = [
  0.00000000e+00
  0.00000000e+00
  8.60726073e-01
  8.51485149e-02
  2.04620462e-02
  1.12211221e-02
  7.92079208e-03
  3.30033003e-03
  3.96039604e-03
  2.64026403e-03
  1.98019802e-03
  0.00000000e+00
  0.00000000e+00
  0.00000000e+00
  1.32013201e-03
  6.60066007e-04
  0.00000000e+00
  0.00000000e+00
  6.60066007e-04
  0.00000000e+00
  0.00000000e+00];

x = 0:0.05:1;
H2 = H2./sum(H2);

figure;
bar(x, H2)
xlabel('Difference Ratio')
ylabel('Percentage of merchants')