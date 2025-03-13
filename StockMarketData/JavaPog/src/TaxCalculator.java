import java.util.List;
public class TaxCalculator {
    // Declare instance variables
    private String name;
    private int quantity;
    private Double priceBought;
    private List<Double> priceSold;
    private int  totalProfit;

    // Constructor
    public TaxCalculator(String name, int quantity, Double priceBought, List<Double> priceSold, int totalProfit) {
        this.name = name;
        this.quantity = quantity;
        this.priceBought = priceBought;
        this.priceSold  = priceSold;
        this.totalProfit = totalProfit;
    }
      
    public String getName() {
        return this.name;
    }

    public int getTotalProfit() {
        return this.totalProfit;
    }

    public void buyMoreStock(int additionalQuantity, double newPriceBought) {
        this.priceBought = ((this.quantity * this.priceBought) + (additionalQuantity * newPriceBought)) 
                            / (this.quantity + additionalQuantity);
        this.quantity += additionalQuantity;
    }
    public Double CalculateProfit(int index, int quantity) {
        this.totalProfit +=(this.priceSold.get(index)-this.priceBought)*quantity;
        return (this.priceSold.get(index)-this.priceBought)*quantity;
    }
    public TaxCalculator StockData() {
        System.out.println(this.CalculateProfit(0, 500));
        this.buyMoreStock(200, 10.10);
        this.buyMoreStock(210, 10.22);
        this.buyMoreStock(90, 10.59);
        System.out.println(this.CalculateProfit(1, 250));
        this.buyMoreStock(75, 10.51);
        this.buyMoreStock(100, 11.38);
        System.out.println(this.CalculateProfit(2,425));
        this.buyMoreStock(300, 10.47);
        System.out.println(this.CalculateProfit(3,150));
        this.buyMoreStock(250, 9.90);
        System.out.println(this.CalculateProfit(4,300));
        this.buyMoreStock(300, 8.87);

        return this;
    }
    public static void main(String[] args) {
        TaxCalculator taxCalculator = new TaxCalculator("CLSK", 500,9.53, 
        List.of(10.26, 10.62, 10.81,9.89,8.85,8.32),0);
        taxCalculator.StockData();
        System.out.println(taxCalculator.getTotalProfit()*7.12+2608.0+1336.0+1908-9100 + " "+ "DKK");
        System.out.println(taxCalculator.getName());
    }
    
}

