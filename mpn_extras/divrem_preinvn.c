/*=============================================================================

    This file is part of FLINT.

    FLINT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    FLINT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FLINT; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

=============================================================================*/
/******************************************************************************

    Copyright (C) 2013 William Hart

******************************************************************************/

#include <stdlib.h>
#include <gmp.h>
#include "flint.h"
#include "longlong.h"
#include "mpn_extras.h"

void flint_mpn_divrem_n_preinvn(mp_ptr q, mp_ptr a, mp_size_t m,
                          mp_srcptr d, mp_size_t n, mp_srcptr dinv)
{
   mp_limb_t cy;
   mp_ptr t;
   mp_size_t size = m - n;
   TMP_INIT;

   TMP_START;
   t = TMP_ALLOC(3*n*sizeof(mp_limb_t));
   
   if (size)
   {
      mpn_mul(t + n, dinv, n, a + n, size);
      cy = mpn_add_n(q, t + 2*n, a + n, size);

      mpn_mul(t, d, n, q, size);
      if (cy && m < 2*n)
         mpn_add_n(t + size, t + size, d, n);
      
      cy = a[n] - t[n] - mpn_sub_n(a, a, t, n);

      while (cy > 0)
      {
         cy -= mpn_sub_n(a, a, d, n);
         mpn_add_1(q, q, n, 1);
      }
   } else
      mpn_zero(q, size);

   if (mpn_cmp(a, d, n) >= 0)
   {
      mpn_sub_n(a, a, d, n);
      mpn_add_1(q, q, n, 1);
   }

   TMP_END;
}

mp_limb_t flint_mpn_divrem_preinvn(mp_ptr q, mp_ptr a, mp_size_t m, 
                                      mp_srcptr d, mp_size_t n, mp_srcptr dinv)
{
   mp_limb_t hi = 0;
   
   if (mpn_cmp(a + m - n, d, n) >= 0)
   {
      mpn_sub_n(a + m - n, a + m - n, d, n);
      hi = 1;
   }
      
   while (m > 2*n)
   {
      flint_mpn_divrem_n_preinvn(q + m - 2*n, a + m - 2*n, 2*n, d, n, dinv);
      m -= n;
   }

   flint_mpn_divrem_n_preinvn(q, a, m, d, n, dinv);

   return hi;
}

