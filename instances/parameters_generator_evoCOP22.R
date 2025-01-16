# Instances parameters generator using LHS sampling
#
# Authors: LEC Group, LISIC Lab, ULCO, France
# date: 2021/08/16

require(lhs)

param_generator <- function(n_min = 14, n_max = 14, nInstances = 50, s = 1000, filename = "puboi_param_1000seed.csv", deg1_value = 2) {
  # Output file
  fil = file(filename)
  
  # bounds of problem dimension
#  n_min <- 1000
#  n_max <- 5000
  # maximum degree of importance
  deg_max <- 10
  # Number of terms (clauses), proportion of maximum number
  # m [0.01, 0.2] * n*(n-1)/2
  m_min <- 0.01
  m_max <- 0.2
  # first id of the instances set
  id <- s
  # Number of instances
#  nInstances <- 1000
  # Random seed
#  s <- 0
  # Type of weigth for each clause. 0: one (equal), 1: Geometric mean of importance
  typeWeight <- 0
  # Random shuffle of the maximum of each clauses
  shift <- 1  # 0: False, optimum=1111 ; 1:True optimum=shuffle()
  # Number of class of importance
  n_class <- 2 # two classes : important, and not important variables
  # default degree of importance for each class
  deg <- rep(1, n_class) 
  #print(deg)

  # Latin Hypercube Sampling of parameters
  set.seed(s)
  sampleSize <- 4 * nInstances # 4 variables per function 
  nFeatures <- 7
  sample <- randomLHS(sampleSize, nFeatures)
  nInstances_deg2 <- nInstances / 2  # Adjust as needed

  res <- c("id type n n_class size degree factor typeWeight n_p p m shift seed")

  i <- 1  
  for(j in 1:nInstances) {
    p <- sample[i, 1:3]
    while (sum(p) > 1) {
      if (i < nrow(sample))
        i <- i + 1
      else {
        sample <- randomLHS(sampleSize, nFeatures)
        i <- 1
      }
      p <- sample[i, 1:3]
    }
    p <- c(p, 1 - sum(p))
    #p <- p / sum(p)
    
    # Only set up for 2 classes of importance 
    n <- n_min + round(sample[i,4] * (n_max - n_min))
    n1 <- round(0.25 * n)
    n2 <- n - n1
   # print(sample)
    deg[2] <- 1 # keep this t he same 
    # Set degrees of importance
    deg[2] <- 1  # Keep this the same
    deg[1] <- deg1_value  # Set deg[1] to the specified value (2 or 10)
      
      deg[2] <- 1  # Keep deg[2] as 1
    #deg[1] <- 2
    #deg[1] <- 10
    p0 <- deg[1] / sum(deg)
    
    f <- 1 + sample[i,6] * (1/p0 - 1) # factor?
    
    m_rate <- m_min + sample[i,7] * (m_max - m_min)
   # m <- round(m_rate * n * (n - 1) / 2) # sub functions?? - start with 5 and see how we get on
    m <- 5 # change to 10 if necessary

    # pretty print    
    line <- paste(id, "puboi", n, n_class)
    line <- paste(line, ' "', n1, ",", n2, '"', sep = '')
    line <- paste(line, ' \"', format(round(deg[1], 6), nsmall = 6), ",", deg[2], '\"', sep ='')
    line <- paste(line, format(round(f, 6), nsmall = 6), typeWeight)
    line <- paste(line, ' 4 \"', format(round(p[1], 6), nsmall = 6), ",", format(round(p[2], 6), nsmall = 6), ",", format(round(p[3], 6), nsmall = 6), ",", format(round(p[4], 6), nsmall = 6), '\"', sep = '')
    line <- paste(line, ' ', m, ' ', shift, ' ', s, sep = '')
    
    cat(line, '\n')
    res <- c(res, line)

    id <- id + 1
    i <- i + 1
  }
  
  writeLines(res, con = fil)
  close(fil)
}

main <- function() {
   setwd('C:\\Users\\40283127\\OneDrive - Edinburgh Napier University\\University\\PhD\\pubo_importance_benchmark\\instances')
   # C:\Users\40283127\OneDrive - Edinburgh Napier University\pubo_importance_benchmark\instances

  #param_generator(n_min = 14, n_max = 14, nInstances = 100, s = 1000, filename = "small\\puboi_param_1000seed.csv") # 100, 50 for the extreme and not extreme cases, then these need to be pruned to only get instances that contain variables of important in the terms, for ones that do not contain even one of the terms, we reject these and move onto evaluating the next instance. 
    # Generate 50 instances for deg[1] = 2
   param_generator(n_min = 14, n_max = 14, nInstances = 50, s = 1000, filename = "small\\puboi_param_1000seed_deg2.csv", deg1_value = 2)
    
    # Generate 50 instances for deg[1] = 10
   param_generator(n_min = 14, n_max = 14, nInstances = 50, s = 1050, filename = "small\\puboi_param_1000seed_deg10.csv", deg1_value = 10)

   
 # Examples used by the authors
  
  # small example to test
   #param_generator(n_min = 50, n_max = 100, nInstances = 5, s = 1000, filename = "small/puboi_param_small.csv")
  
  # parameters of evoCOP 2022
   #param_generator(n_min = 1000, n_max = 5000, nInstances = 1000, s = 0, filename = "puboi_param.csv")



  # smaller number of class variables - as requested by Sarah
    # small example to test
# do 30 instances instead of 5, central limit thereon
# doing 10 as the minimum raises an index error for the terms list (so for[0,1][1,2], [0,1][]) <- if the second list is empty then the check cannot be done and then this raises an index error 
  # 14 is the number it works with, need to discuss this with Sarah
  
  # larger number of instances
  #param_generator(n_min = 14, n_max = 14, nInstances = 1000, s = 0, filename = "puboi_param_class.csv")
}

main()
