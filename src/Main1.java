import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

//Solution to the question from : Bytedance
//Z国的货币系统包含面值1元、4元、16元、64元共计4种硬币，以及面值1024元的纸币。现在小Y使用1024元的纸币购买了一件价值为的商品，请问最少他会收到多少硬币？
public class Main1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(System.in);
		int price = sc.nextInt();
		int res = 0;
		List<Integer> coins = new LinkedList<Integer>();
		coins.add(64);
		coins.add(16);
		coins.add(4);
		coins.add(1);
		Iterator<Integer> iter = coins.iterator();
		while(iter.hasNext()){
			int temp = iter.next();
			res += price/temp;
			price = price%temp;
		}
		System.out.println(res);
	}

}
