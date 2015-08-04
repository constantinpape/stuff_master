#pragma once
#include <vector>
#include <utility>
#include <exception>
#include <algorithm>
#include <iterator>

template<class ITERATOR_0, class ITERATOR_1>
std::pair<double, double> fscore(
        ITERATOR_0 segA_begin,
        ITERATOR_0 segA_end,
        ITERATOR_1 segB_begin,
        ITERATOR_1 segB_end
        )
{
    typedef typename std::iterator_traits<ITERATOR_0>::value_type Label0;
    typedef typename std::iterator_traits<ITERATOR_1>::value_type Label1;

    size_t N = std::distance(segA_begin, segA_end);

    if ( N != std::distance(segB_begin, segB_end) )
        throw std::runtime_error("Segmentation sizes do not match!");

    size_t N_labels_A = *( std::max_element( segA_begin, segA_end ) );
    size_t N_labels_B = *( std::max_element( segB_begin, segB_end ) );

    // init the contingency matrix 
    std::vector<size_t> init_row(N_labels_B + 1, 0);
    std::vector< std::vector<size_t> > p_ij( N_labels_A + 1, init_row );

    // compute the contingency matrix
    ITERATOR_0 segA_it = segA_begin;
    ITERATOR_1 segB_it = segB_begin;
    for( ; segA_it != segA_end; segA_it++ )
    {
        Label0 i = *( segA_it );
        Label1 j = *( segB_it );
        p_ij[i][j] += 1;
        segB_it++;
    }

    // compute the sum of rows
    std::vector<double> a_i(p_ij.size(), 0.);
    for( size_t i = 1; i < a_i.size(); i++ )
    {
        for( size_t j = 0; j < p_ij[0].size(); j++ )
        {
            a_i[i] += p_ij[i][j];
        }
    }
    
    // compute the sum of cols
    std::vector<double> b_j(p_ij.size(), 0.);
    for( size_t j = 1; j < b_j.size(); j++ )
    {
        for( size_t i = 0; i < p_ij.size(); i++ )
        {
            b_j[j] += p_ij[i][j];
        }
    }

    double aux = 0.;
    for( size_t i = 1; i < p_ij.size() ; i++)
    {
        aux += p_ij[i][0];
    }

    // sum of square of rows
    double sumA = 0.;
    for( size_t i = 0; i < a_i.size(); i++ )
    {
        sumA += a_i[i] * a_i[i];
    }
    
    // sum of square of cols
    double sumB = 0.;
    for( size_t j = 0; j < b_j.size(); j++ )
    {
        sumB += b_j[j] * b_j[j];
    }

    sumB += aux / N;

    double sumAB = 0.;
    for( size_t i = 1; i < p_ij.size(); i++ )
    {
        for( size_t j = 1; j < p_ij[0].size(); j++ )
        {
            sumAB += p_ij[i][j]*p_ij[i][j];
        }
    }

    sumAB += aux / N;

    double prec = sumAB / sumB;
    double rec  = sumAB / sumA;
    double ri = 1. - (sumA + sumB - 2.0*sumAB) / (N*N);
    double f_score = 2.0 * prec * rec / (prec + rec );

    return std::pair<double, double>(ri, f_score);
}

