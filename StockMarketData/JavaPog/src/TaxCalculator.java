import java.util.List;
public class TaxCalculator {
    // Declare instance variables
    private String name;
    private int quantity;
    private Double priceBought;
    private List<Double> priceSold;
    private double totalProfit;
    private int fee = 3;
    
    // Constructor

    public TaxCalculator(String name, int quantity, Double priceBought, List<Double> priceSold, double totalProfit) {
        this.name = name;
        this.quantity = quantity;
        this.priceBought = priceBought;
        this.priceSold  = priceSold;
        this.totalProfit = totalProfit;
    }

    public String getName() {
        return this.name;
    }


    public double getTotalProfit() {
        return this.totalProfit;
    }
    public void buyMoreStock(int additionalQuantity, double newPriceBought) {
        this.priceBought = ((this.quantity * this.priceBought) + (additionalQuantity * newPriceBought)+this.fee) 
                            / (this.quantity + additionalQuantity);
        this.quantity += additionalQuantity;
    }
    public Double CalculateProfit(int index, int quantitySold) {
        this.totalProfit += (this.priceSold.get(index)-this.priceBought)*quantitySold-this.fee;
        this.quantity-=quantitySold;
        return (this.priceSold.get(index)-this.priceBought)*quantitySold-this.fee;
    }
    public TaxCalculator StockDataNVO() {
        this.CalculateProfit(0,66);
        //System.out.println(this.CalculateProfit(0,66));
        //System.out.println(this.priceBought);
        this.buyMoreStock(45, 81.55);
        this.buyMoreStock(16, 87.91);
        this.CalculateProfit(1,61);
        //System.out.println(this.CalculateProfit(1,61));
        System.out.println(this.getTotalProfit());
        return this;
    }

    public TaxCalculator StockDataCLSK() {
        this.CalculateProfit(0, 500);
        this.buyMoreStock(200, 10.10);
        this.buyMoreStock(210, 10.22);
        this.buyMoreStock(90, 10.59);
        //System.out.println(this.priceBought);
        this.CalculateProfit(1, 250);
        this.buyMoreStock(75, 10.51);
        this.buyMoreStock(100, 11.38);
        this.CalculateProfit(2,425);
        this.buyMoreStock(300, 10.47);
        this.CalculateProfit(3,150);
        this.buyMoreStock(250, 9.90);
        this.CalculateProfit(4,300);
        this.buyMoreStock(300, 8.87);
        System.out.println(this.getTotalProfit());
        return this;
    }



    public static void main(String[] args) {
        TaxCalculator taxcalculatorCLSK = new TaxCalculator("CLSK", 500,9.53, 
        List.of(10.26, 10.62, 10.81,9.89,8.85,8.32),0);
        taxcalculatorCLSK.StockDataCLSK();
        //System.out.println(taxcalculatorCLSK.getTotalProfit());
        //System.out.println(taxcalculatorCLSK.getName());

        TaxCalculator taxCalculatorNVO = new TaxCalculator("NVO", 66,80.66, 
        List.of(83.03, 86.7),0);
        taxCalculatorNVO.StockDataNVO();
        //taxCalculatorNVO.getTotalProfit();
        int PLTR = 2037;
        System.out.println((taxCalculatorNVO.getTotalProfit()+taxcalculatorCLSK.getTotalProfit())*7+2608+1910-9100 + PLTR + " " +"DKK of profit");
    }

}