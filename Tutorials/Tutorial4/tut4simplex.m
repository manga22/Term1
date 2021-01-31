%CMO 2020 Tutorial 4 simplex example

%% Set up the problem data

A = [1 0 1 0 0;...
    0 1 0 1 0;...
    2 3 0 0 1]
b= [2;1;6]
c = [-1 -1 0 0 0]
pause
I = eye(5);
e1 = I(:,1);
e2 = I(:,2);
e3 = I(:,3);
e4 = I(:,4);
e5 = I(:,5);
%% Step 1: Find the BFS and solve for x_B

B = A(:,1:3);
N = A(:,4:5);
c_B = c(1:3).'
c_N = c(4:5).'


x_B = inv(B)*b;
x_N = zeros(2,1);

%Optimality check:
c_test = c - c_B.'*inv(B)*A;

%Dual variables
p = inv(B.')*c_B;
s_B = zeros(3,1);
s_N = c_N - N.'*p;
%s_N has a negative value, so the BFS isn't optimal. What next?

%% Next step

%From the previous step, we see that c_test(4) = -.5. So we'll pick 

j= 4;
Aj = inv(B)*A(:,j);

t = Aj.\x_B;
alpha = min(t(find(t>0)));

x_new = [x_B-alpha*Aj;x_N]+alpha*e4 ;%+ alpha*(e4-[Aj;x_N]);

%% Here we go again

basis = [1 2 4];
B = A(:,basis);
c_B = c(basis).';
c_N = c([3,5]);

%optimality?
c_test = c - c_B.'*inv(B)*A;
x_opt = inv(B)*b;
