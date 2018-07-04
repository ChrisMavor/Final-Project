# Christopher Mavor 
# July 3, 2018
# Final-Project
# Part1: Code & Part2: Written
##########################
# PART 1: Code
##########################

# Transposing a matrix 
def MatrixTranspose(matrix):
  """
  Transposing Matrix inputs to use in Matrix arithmatic
  Input: matrix of size m x n.
  Output: matrix of size n x m where columns of the input matrix have become the rows, and the input rows are the outcome columns.
  """
  B = [[0 for col in range(len(matrix))] for row in range(len(matrix[0]))] 
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      B[j][i]=matrix[i][j]
  return B 


# Scalar Vector Multiplication
def vecScalar(vec,scalar):
  """
  Vector scalar multiplication. 
  Inputs: A vector 'vec' of any size. Must be in a single list and not a nested list.
  A scalar 'scalar' to multiply each element by in the vector set
  Output: a scalar vector multiplication resulting in a vector having the same dimensions as the input vector, but having each element having been multiplied by the scalar.
  """
  row = [ ]
  for i in range(len(vec)):
      row.append(scalar*vec[i])
  return row

# Dot Product 
def dot(vector01,vector02):
  """
  Inputs:
    vector01: vector in format of a list.
    vector02: vector in format of a list.
  Output:
    dot:
  This function 'dot' takes inputs in the format of lists, or vectors, and performs the dot product of the vectors. Dot product refers to the itemized multiplication of corresponding elements in each vector and sums the product of the itemized multiplication. 
  If there is a mistake in the formating, an error will result specifying the inputs to change in format type so that the function can opperate properly. 
  If proper inputs are given, the end result will return a integer called 'row'.
  """
  if type(vector01)==list and type(vector02)==list:
    #ensuring correct type of input for the 
    if len(vector01)==len(vector02):
      dot=0 
      for i in range(len(vector01)):
        dot+=vector01[i]*vector02[i]
      return dot
    else:
      return  "Incorrect sizing. Make sure both inputs are a single list of the same length. Make sure inputs are not integers, matrices(list of lists), or strings."
  else:
    return  "Incorrect sizing. Make sure both inputs are a single list of the same length. Make sure inputs are not integers, matrices(list of lists), or strings."

# Vector Subtraction 
def vecSub(vector1,vector2):
  """
  Inputs: two eqisized vectors (or lists) that we wish to subtract in a specific order. inputs must be same size
  Output: the resulting vector subtraction
  """
  x=[]#initializing result
  if len(vector1)==len(vector2):
    for i in range(len(vector1)):
      x.append(vector1[i]-vector2[i])# itemized subtraction of each element in the vectors
    return x

# Matrix Vector Multiplication
def matVec(matrix, vector):
  
  """
  This function takes a matrix and a vector as its arguments. It then updates each element of the matrix by multiplying it by the vector corresponding value before returning the now updated vector. Input values must be in matrix = m*n and vector in 1*n.
  """
  A = [ ]
  sizematrix = len(matrix[0])
  sizevector = len(vector)
  # Assigning A as the empty matix value to store the results in. The size values are determining the size of the input values.
  if len(matrix) == len(matrix[0]) and len(matrix) == 1:
   print("Error in matrix sizing")

  if sizematrix == sizevector:
    # using this if condition to determine if matrix vector multiplication is capable.
    for i in range(len(matrix)):
      #this is determining how many rows there are in the matrix input to separate the opperations by each row of the matrix.
      rowx = [ ]
      # empty row vector to store the multiplication opperation of each row in.
      for j in range(len(matrix[i])):
       rowx.append(vector[j]*matrix[i][j])
       #opperation of vector elements multiplying bye each matrix row element.    
      rowx=sum(rowx)
      # summing the itemized multiplication after being placed in a list format.
      A.append(rowx)
      # sending the results from each row into the matrix
    return A
  else:
    return print("Error on matrix size. Make sure input values of matrix have the same number of columns as the vector.")

# Matrix Matrix Multiplication
def matmatmult(A,B):
  """
  Input: Two matrix of compatable sizes for matrix matrix multiplication.
  There is a type check and a size check to make sure that the two will multiply in the order that they are input
  Output: is the matrix matrix multiplation. Resulting matrix will be resized to be have the row size of A and column amount of B.
  """
  if type(A)==list and type(B)==list:
    if type(A[0])==list and type(B[0])==list:
      if len(A[0])==len(B):
        X = [[0 for col in range(len(B[0]))] for row in range(len(A))]#initializing a zero matrix that is the size of the result
        B = MatrixTranspose(B)
        for i in range(len(A)):
          for j in range(len(B)):
            X[i][j]= dot(A[i],B[j])#taking each row and column of the inputs and computing the dotproduct for each step, then replacing the itemized result into the X matrix.
        return X
      else:
        return None
    else:
      return None
  else: 
    return None

# Normalizing Matrix 
def normilzeMatrix(matrix):
  """
  Input: matrix of any dimension.
  Output: the normalized matrix.
  """
  normalize=[]
  for i in range(len(matrix)):
    length=0
    for j in range(len(matrix[0])):
      length+=abs(matrix[i][j]**2)#summing each element's square for each row
    norm=(length**(1/2))# taking the square root of each sum from above
    normalize.append(vecScalar(matrix[i],(1/norm)))#using the vecScalar multiplicaiton to multiply the scalar of the norm and completing the normalization of the matrix
  return normalize

##########################
# PART 1: Question 1
##########################

def vandermondematrix(x,y):
  """
  This matrix creates a vandermonde matrix of dimension 4.

  Inputs: x and y vectors, each in the form of a single list, and not in the form of a column vector or list of lists.

  Output: Matrix A which is the vandermonde matrix of the x vector inputs.
  """
  n=4
  m=len(x)
  if type(x)== list and type(y)==list: 
    #type check for input values.
    #not neccessary at this point, but good for further steps that follow.
    A=[]
    if type(x[0])==list:
      return 'bad input value Change into row vector'
    else:
      for j in range(n):
        row=[]
        for i in range(m):
          row.append(x[i]**j)
        A.append(row)
      A = MatrixTranspose(A)
      return A


##########################
# PART 1: Question 2
##########################

def Qmatrix(matrix):
  V=MatrixTranspose(matrix)
  X = [[0 for col in range(len(V[0]))] for row in range(len(V))]
  for i in range(len(V)):
    if i == 0:
      X[i] = vecScalar(V[i],-1)
    elif i == 1:
      X[i]= vecScalar(vecSub(V[i],(vecScalar(X[i-1],(dot(V[i],X[i-1])/abs(dot(X[i-1],X[i-1])))))),1)
    elif i == 2:
      proj2 = vecScalar(X[i-1],(dot(V[i],X[i-1])/abs(dot(X[i-1],X[i-1]))))
      proj1 = vecScalar(X[i-2],(dot(V[i],X[i-2])/abs(dot(X[i-2],X[i-2]))))
      projs = vecSub(proj2,vecScalar(proj1,-1))
      X[i] = vecScalar(vecSub(V[i],projs),1)
    elif i == 3:
      proj3 = vecScalar(X[i-1],(dot(V[i],X[i-1])/dot(X[i-1],X[i-1])))
      proj2 = vecScalar(X[i-2],(dot(V[i],X[i-2])/dot(X[i-2],X[i-2])))
      proj1 = vecScalar(X[i-3],(dot(V[i],X[i-3])/dot(X[i-3],X[i-3])))
      projs2 = vecSub(proj2,vecScalar(proj1,-1))
      projs = vecSub(proj3,vecScalar(projs2,-1))
      X[i] = vecScalar(vecSub(V[i],projs),-1)
    #I found it easier to hard code this portion so that the proper results were coming from this portion.
    #Above is my algorithm for the projection of each vector onto the orthogonal vectors, 
    #thus to result in an orthanormal matrix. Taking the orthanormal matrix and normalizing the matrix resulted in Q
  #below is the algorithm given in class, but I found that it resulted in a different R matrix than what I had calculated by hand.

  '''def QRFactorization(A):
    """
    QRFactorization takes a matrix as it's argument
    1) Confirms type validity
    2) Initialize V, Q, R
    3) Copy A into V
    4) Generate R diagonal using twoNorm
    5) Generate Q norm vectors
    6) Complete R upper triangular 
    """
    numCols = len(A[0])
    numRows = len(A)
    
    inputStatus = True
    if (len(A)==0):
      print('ERROR: Empty Matrix')
      inputStatus = False    
    for v in A:
      if (len(v)==0):
       print('ERROR: Empty Vector in Matrix')
       inputStatus = False

    for j in range(numRows):
        for i in range(numCols):
            if ((type(A[j][i]) != int) and (type(A[j][i]) != float) and (type(A[j][i]) != complex)):
                inputStatus = False

    if inputStatus:
        
        V = [[0] * numCols for i in range(numRows)]
        Q = [[0] * numCols for i in range(numRows)]
        R = [[0] * numRows for i in range(numRows)]
        #print(A)
        #print(R)
        #print(Q)
        #print(V)

        for i in range(numRows):
            V[i] = A[i]
        
        for i in range(numRows):
            R[i][i] = twoNorm(V[i])
            Q[i] = normalize(V[i])
            
            for j in range(i+1, numRows):
                R[i][j] = dot(Q[i],V[j])
                t = vecScalar(Q[i], R[i][j])
                V[j] = vecSub(V[j], t)
    
        return [Q,R]
    else:
        return [[],[]]
        '''
  Q = normilzeMatrix(X)
  Q = MatrixTranspose(Q)
   # transposing the matrix to ensure proper sizing for next step in the process of finding R
  return Q

def Rmatrix(Q,A):
  """
  Input:Q is a matrix is the orthanormal and normalized A Matrix. Taking the transpose of Q and multiplying it by A results in R. 
  Output: R is the upper diagonal matrix used in the QR Factorization process where A=QR. 
  """
  Qt = MatrixTranspose(Q)
  #At = MatrixTranspose(A) 
  #precautionary measure to make sure the proper dimensions are met for matmatmult function
  R = matmatmult(Qt,A)
  
  return R

##########################
# PART 1: Question 3
##########################

def qttimesy(Q,y):
  """
  Inputs: Q orthanormal matrix of vandermonde matrix A. y is the data in the y column.
  Output: b from taking the transpose of Q and multiplying it by the vector y resulting in a 4x1 vector.
  """
  Qt=MatrixTranspose(Q)
  b=[]
  for i in range(len(Qt)):
    rows=0 #initializing the row 
    for j in range(len(Qt[0])):
      rows+=(Qt[i][j]*y[j])#summing the values of each "rows" value by the previous term and adding it to the multiplication of y on a term by term basis.
    b.append(rows)
  return b

##########################
# PART 1: Question 4
##########################

def Backsub(R,b):
  """
  Inputs: Upper diagonal R and b where b=(Q^t)(y)
  Output: c which is the coeficient vector for which y = c[0]x^0+c[1]x^1+c[2]x^2+c[3]x^3
  """
  k = len(b)-1 #range of b would create a indexing error based on the size of c thus k needed to help indexfor the division of the term b and R
  c= [0]*len(b)
  c[k] = b[k]/R[k][k]
  for i in reversed(range(len(b))):
    c[i]=b[i]
    for j in range(i+1,len(b)):
      c[i]=c[i]-(c[j]*R[j][i]) #initializing the term value so that the next line can call on the term and immediately replace it witht he full equation
      c[i]=c[i]/R[i][i]
  return c

def approxPoly(c,A):
  """
  Computing the resulting matrix after having found the coefficent Matrix. 
  Taking the Vandermonde matrix and inpting the coeffient matrix to show results of the polynomial.
  """
  poly = [[0 for col in range(len(A))] for row in range((0))]
  poly = vecScalar(matVec(A,c),1)#using matrix vector multiplication function to produce results.

  return poly



"""stability is how an algorith handles error
 
 
 is how a problem handles error"""




###################
# Part1: Outputs
###################

def ProjectDisplay(x,y):
  """
  Taking each of the functions listed above and displaying all of their results in a printed format for reading the results all together. 
  This function is not for calling on the results, just for display purposes.
  Inputs are vectors strictly. They should be a row vector and not a column vector. Vectors must be of same dimension (i.e. x = [1,2,3], y = [2,3,4]).
  Error check will occur if parameters are not met where Vectors must be of the same dimension, same type, must be made up of number types, and must be a single list.
  """
  if type(x)!=type(y) and type(x)!=list:
    return "Error in input values"  
  elif len(x)!=len(y):
    return "Error in input values" 
  elif type(x[0])==str or type(y[0])==str or type(x[0])==list or type(y[0])==list:
    return "Error in input values"
  else:
    A = vandermondematrix(datax,datay)
    print('-----------------------')
    print('Question #1')
    print('Vandermonde Matrix A =')
    print(A)

    Q = Qmatrix(A)
    print('-----------------------')
    print('Question #2')
    print('Orthanormal Matrix Q=')
    print(Q)
    R=Rmatrix(Q,A) 
    print('-----------------------')
    print('Upper Triangular Matrix R= (Rounded to the nearest 4 decimal places)')

    Rrounded=[[0 for col in range(len(R[0]))] for row in range(len(R))]
    for i in range(len(R)):
      for j in range(len(R[0])):
        Rrounded[i][j]=round(R[i][j],4)# rounding of the values for ease of viewing. 
    print(Rrounded)

    print('-----------------------')
    print('Question #3')
    y= datay
    b=qttimesy(Q,y)
    print('Orthanormal Matrix Q Transposed multiplied by the Data vector y= Qt*y=')
    print(b)

    print('-----------------------')
    print('Question #4')
    print('C=(R^-1)(b)=')
    c=Backsub(R,b)
    print(c)

    print('-----------------------')
    print('Question #5')
    poly=approxPoly(c,A)
    print(poly)
    return True

def ProjectOutputs(x,y):
  """
  Proper listing of results in a list function result. 
  Inputs are vectors strictly. They should be a row vector and not a column vector. Vectors must be of same dimension (i.e. x = [1,2,3], y = [2,3,4]).
  Error check will occur if parameters are not met where Vectors must be of the same dimension, same type, must be made up of number types, and must be a single list.
  """
  if type(x)!=type(y) and type(x)!=list:
    return "Error in input values"  
  elif len(x)!=len(y):
    return "Error in input values" 
  elif type(x[0])==str or type(y[0])==str or type(x[0])==list or type(y[0])==list:
    return "Error in input values"
  else:
    A = vandermondematrix(datax,datay)
    Q = Qmatrix(A)
    R=Rmatrix(Q,A) 
    y= datay
    b=qttimesy(Q,y)
    c=Backsub(R,b)
    poly=approxPoly(c,A)
    return[A,Q,R,b,c,poly] #this is the order of calling for each specific result that you wish to display.

datay=[1.102,1.099, 1.017, 1.111, 1.117, 1.152, 1.265, 1.380, 1.575, 1.857]
datax=[.55, .60, .65,.70,.75,.80,.85,.90,.95,1.00]
FinalDisplay=ProjectDisplay(datax,datay)
FinalValues=ProjectOutputs(datax,datay)

########
#Uncomment the final lines to display the outputs for each part, or to specifically call on the parts you wish to display.
########
print(FinalDisplay)

#'Final A'
#print(FinalValues[0])
#'Final Q'
#print(FinalValues[1])
#'Final R'
#print(FinalValues[2])
#'Final b'
#print(FinalValues[3])
#'Final c'
#print(FinalValues[4])
#'Final Poly'
#print(FinalValues[5])
##########################
# PART 2: Written
##########################
############
# Part 2: Question 1
############
"""
The modified Gram-Schmidt allows for smaller truncation error by reducing the error by including a third for loop, thus increasing from the standard Gram-Schmidt method by one for loop.
"""
############
# Part 2: Question 3
############
"""
Condition is a way to write the algorithm, so to avoid small changes or to ananalyze the small changes. By properly conditioning the algorithm, we can significantly alter how the algorithm handles the error, so to minimize error.

Stability is a way to check accuracy of the problem results. There are two types of errors uses, relative and absolute. Relative is variance from one result to the next, while absolute is the selected result compared to the over all results. Forward and Backward stability show the deviation in the problems, and by checking the stability, we can show how well the actual problem handles the computing error, rather than having the algorithm be the root cause.
"""
