class Solution:
    def romanToInt(self, s: str) -> int:
        dictionairy = {"I":1, "V":5, "X":10, "L":50, "C": 100, "D":500, "M":1000}
        str_new = list(s)
        sum = 0
        for char in str_new:
            if char in dictionairy.keys():
                sum+=dictionairy[char]

        if "IV" in s:
            sum-=2
        
        if "IX" in s:
            sum-=2
        
        if "XC" in s:
            sum-=20

        if "XL" in s:
            sum-=20
        
        if "CD" in s:
            sum-=200

        if "CM" in s:
            sum-=200

        return sum


