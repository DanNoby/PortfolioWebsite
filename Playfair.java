import java.util.*;
import java.math.*;

// PLAYFAIR CIPHER
public class Playfair {
    public static String encryptPair(String pair, String table) {
        int a = table.indexOf(pair.charAt(0));
        int b = table.indexOf(pair.charAt(1));
        int r1 = a / 5, c1 = a % 5, r2 = b / 5, c2 = b % 5;
        if (r1 == r2)
            return "" + table.charAt(r1 * 5 + (c1 + 1) % 5) + table.charAt(r2 * 5 + (c2 + 1) % 5);
        if (c1 == c2)
            return "" + table.charAt(((r1 + 1) % 5) * 5 + c1) + table.charAt(((r2 + 1) % 5) * 5 + c2);
        return "" + table.charAt(r1 * 5 + c2) + table.charAt(r2 * 5 + c1);
   }

   public static String decryptPair(String pair, String table) {
       int a = table.indexOf(pair.charAt(0));
       int b = table.indexOf(pair.charAt(1));
       int r1 = a / 5, c1 = a % 5, r2 = b / 5, c2 = b % 5;
       if (r1 == r2)
           return "" + table.charAt(r1 * 5 + (c1 + 4) % 5) + table.charAt(r2 * 5 + (c2 + 4) % 5);
       if (c1 == c2)
           return "" + table.charAt(((r1 + 4) % 5) * 5 + c1) + table.charAt(((r2 + 4) % 5) * 5 + c2);


       return "" + table.charAt(r1 * 5 + c2) + table.charAt(r2 * 5 + c1);
   }
   public static void main(String[] args) {
        String text = "INFORMATIONS";
        String key = "NETWORK";
        
        // table gen
        String table = "";
        String alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ";
        String combined = (key + alphabet).toUpperCase().replace("J", "I");
        for (char c : combined.toCharArray()) {
            if (table.indexOf(c) == -1) table += c;
        }

        for(int i=0; i<table.length(); i++) {
            if(i%5==0) System.out.println();
            System.out.print(table.charAt(i));
        }
        System.out.println("\n");

        // pairs
        text = text.toUpperCase().replace("J", "I");
        List<String> pairs = new ArrayList<>();
        for (int i = 0; i < text.length(); i += 2) {
            if (i == text.length() - 1) {
                pairs.add(text.charAt(i) + "X");
            } else if (text.charAt(i) == text.charAt(i + 1)) {
                pairs.add(text.charAt(i) + "X");
                i--;
            } else {
                pairs.add(text.substring(i, i + 2));
            }
        }

        String encrypted = "";
        for (String p : pairs)  {
            encrypted += encryptPair(p, table);
            System.out.println(p);
        }
        System.out.println("Ciphertext: " + encrypted);

        String decrypted = "";
        for (int i = 0; i < encrypted.length(); i += 2) {
            decrypted += decryptPair(encrypted.substring(i, i + 2), table);
        }
        System.out.println("Decrypted:  " + decrypted);
   }
}


// HILL CIPHER

public class HillCipher {
   public static void main(String[] args) {
       System.out.println("Danny Noby Joseph\n23BAI1227\n");
       String plainText = "SUBMITASSIGNMENTINLMS".toUpperCase();
       System.out.println("The plaintext = " + plainText);
       int[][] K = {
           {17, 17, 5},
           {21, 18, 21},
           {2, 2, 19}
       };
      
       // Padding if required
       while (plainText.length() % 3 != 0) plainText += "X";

       // Encryption
       String cipherText = "";
       for (int i = 0; i < plainText.length(); i += 3) {
           cipherText += processBlock(plainText.substring(i, i + 3), K);
       }
       System.out.println("Ciphertext: " + cipherText);

       // somehow get Kinv
       int[][] Kinv = {
           {4, 9, 15},
           {15, 17, 6},
           {24, 0, 17}
       };


       // Decrypt
       String decryptedText = "";
       for (int i = 0; i < cipherText.length(); i += 3) {
           decryptedText += processBlock(cipherText.substring(i, i + 3), Kinv);
       }
       System.out.println("Decrypted:  " + decryptedText);
       
   }

   // This single function handles both because the math steps are identical
   public static String processBlock(String block, int[][] matrix) {
       int[] P = {
           block.charAt(0) - 'A',
           block.charAt(1) - 'A',
           block.charAt(2) - 'A'
       };
       int[] result = new int[3];


       for (int row = 0; row < 3; row++) {
           int sum = 0;
           for (int col = 0; col < 3; col++) {
               sum += matrix[row][col] * P[col];
           }
           result[row] = ((sum % 26) + 26) % 26;
       }


       return "" + (char)(result[0] + 'A') + (char)(result[1] + 'A') + (char)(result[2] + 'A');
   }
}


// RAILFENCE CIPHER

public class RailFence {
    public static String encrypt(String text, int key) {
        char[][] rail = new char[key][text.length()];
        for (char[] row : rail)
            Arrays.fill(row, '\n');

        int row = 0;
        boolean down = false;
        for (int i = 0; i < text.length(); i++) {
            if (row == 0 || row == key - 1)
                down = !down;
            rail[row][i] = text.charAt(i);
            row += down ? 1 : -1;
        }

        String result = "";
        for (char[] r : rail)
            for (char c : r)
                if (c != '\n')
                    result += c;
        return result;
    }

    public static String decrypt(String cipher, int key) {
        char[][] rail = new char[key][cipher.length()];
        for (char[] row : rail)
            Arrays.fill(row, '\n');
        int row = 0;
        boolean down = false;

        for (int i = 0; i < cipher.length(); i++) {
            if (row == 0 || row == key - 1)
                down = !down;

            rail[row][i] = '*';
            row += down ? 1 : -1;
        }

        // fill characters
        int index = 0;
        for (int i = 0; i < key; i++)
            for (int j = 0; j < cipher.length(); j++)
                if (rail[i][j] == '*' && index < cipher.length())
                    rail[i][j] = cipher.charAt(index++);

        // read zigzag
        String result = "";
        row = 0;
        down = false;

        for (int i = 0; i < cipher.length(); i++) {
            if (row == 0 || row == key - 1)
                down = !down;

            result += rail[row][i];
            row += down ? 1 : -1;
        }

        return result;
    }

    public static void main(String[] args) {
        String text = "attack at once";

        String enc = encrypt(text, 2);
        System.out.println("Encrypted: " + enc);

        String dec = decrypt(enc, 2);
        System.out.println("Decrypted: " + dec);
    }
}


// SDES

public class SDES {
    static int[] P10={3,5,2,7,4,10,1,9,8,6};
    static int[] P8 ={6,3,7,4,8,5,10,9};
    static int[] IP ={2,6,3,1,4,8,5,7};
    static int[] IP1={4,1,3,5,7,2,8,6};
    static int[] EP ={4,1,2,3,2,3,4,1};
    static int[] P4 ={2,4,3,1};

    static int[][] S0={{1,0,3,2},{3,2,1,0},{0,2,1,3},{3,1,3,2}};
    static int[][] S1={{0,1,2,3},{2,0,1,3},{3,0,1,0},{2,1,0,3}};

    static String perm(String s,int[] p){
        String r="";
        for(int i:p) r+=s.charAt(i-1);
        return r;
    }

    static String ls(String s,int n){
        return s.substring(n)+s.substring(0,n);
    }

    static String xor(String a,String b){
        String r="";
        for(int i=0;i<a.length();i++)
            r+= (a.charAt(i)==b.charAt(i))?'0':'1';
        return r;
    }

    static String sbox(String s,int[][] box){
        int row=Integer.parseInt(""+s.charAt(0)+s.charAt(3),2);
        int col=Integer.parseInt(""+s.charAt(1)+s.charAt(2),2);
        return String.format("%2s",
            Integer.toBinaryString(box[row][col])).replace(' ','0');
    }

    static String[] keys(String key){
        key=perm(key,P10);
        String l=ls(key.substring(0,5),1);
        String r=ls(key.substring(5),1);
        String k1=perm(l+r,P8);
        l=ls(l,2); r=ls(r,2);
        String k2=perm(l+r,P8);
        return new String[]{k1,k2};
    }

    static String fk(String s,String k){
        String l=s.substring(0,4);
        String r=s.substring(4);
        String t=perm(r,EP);
        t=xor(t,k);
        String s0=sbox(t.substring(0,4),S0);
        String s1=sbox(t.substring(4),S1);
        String p=perm(s0+s1,P4);
        return xor(l,p)+r;
    }

    static String encrypt(String pt,String key){
        String[] k=keys(key);
        pt=perm(pt,IP);
        pt=fk(pt,k[0]);
        pt=pt.substring(4)+pt.substring(0,4);
        pt=fk(pt,k[1]);
        return perm(pt,IP1);
    }

    static String decrypt(String ct,String key){
        String[] k=keys(key);
        ct=perm(ct,IP);
        ct=fk(ct,k[1]);
        ct=ct.substring(4)+ct.substring(0,4);
        ct=fk(ct,k[0]);
        return perm(ct,IP1);
    }

    public static void main(String[] args){
        String key="1010000010";
        String pt="11010111";

        String ct=encrypt(pt,key);
        System.out.println("Cipher: "+ct);
        System.out.println("Decrypted: "+decrypt(ct,key));
    }
}

// RSA
public class RSA{
    public static void main(String[] args){
        int nInt = 143;
        int eInt = 7;
        int pInt = 0, qInt = 0;
        for (int i=2; i<=Math.sqrt(nInt); i++){
            if (nInt%i==0){
                pInt = i;
                qInt = nInt/i;
                break;
            }
        }
        System.out.println("P = "+pInt);
        System.out.println("Q = "+qInt);

        BigInteger n = BigInteger.valueOf(nInt);
        BigInteger e = BigInteger.valueOf(eInt);
        BigInteger p = BigInteger.valueOf(pInt);
        BigInteger q = BigInteger.valueOf(qInt);

        BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));

        BigInteger msg = BigInteger.valueOf(64);

        BigInteger d = e.modInverse(phi);

        BigInteger cipher = msg.modPow(e,n);

        BigInteger decrypted = cipher.modPow(d,n);

        System.out.println("phi = "+phi);
        System.out.println("Msg = "+msg);
        System.out.println("Cipher = "+cipher);
        System.out.println("Decrypted = "+decrypted);
    }
}

// MITM Diffie Hellman

public class MITM{
    public static void main(String[] args){
        BigInteger XA = BigInteger.valueOf(9);//Alice priv
        BigInteger XB = BigInteger.valueOf(14);//Bob priv

        System.out.println("Alice's Priv Key = " +XA);
        System.out.println("Bob's Priv Key = " + XB);

        BigInteger q = BigInteger.valueOf(29);
        BigInteger a = BigInteger.valueOf(2);

        BigInteger XM1 = BigInteger.valueOf(5); //with bob
        BigInteger XM2 = BigInteger.valueOf(11); //with alice

        System.out.println("Middle man's Priv Key(Bob) = " +XM1);
        System.out.println("Middle man's Priv Key(Alice) = " +XM2);


        // computing public keys

        BigInteger YA = a.modPow(XA,q);
        BigInteger YB = a.modPow(XB,q);

        System.out.println("Alice's Pub Key = " +YA);
        System.out.println("Bob's Pub Key = " +YB);


        //middle man's public keys

        BigInteger YM1 = a.modPow(XM1,q);
        BigInteger YM2 = a.modPow(XM2,q);

        System.out.println("MM's Pub Key(Bob) = " +YM1);
        System.out.println("MM's Pub Key(Alice) = " +YM2);

        //Alice n Bob's Secret Key
        BigInteger K = YA.modPow(XB,q);

        System.out.println("Alice & Bob's Secret Key = " +K);

        //Alice n Middleman's Secret Key
        BigInteger K2 = YM2.modPow(XA,q);
        System.out.println("Alice & MM's Secret Key = " +K2);


        //Bob n Middleman's Secret Key
        BigInteger K1 = YM1.modPow(XB,q);

        System.out.println("Bob & MM's Secret Key = " +K1);

    }
}

// Multiuser Diffie Hellman

public class MultiUserDH {
    public static void main(String[] args) {
        // priv keys
        BigInteger XA = BigInteger.valueOf(6);  // Alice
        BigInteger XB = BigInteger.valueOf(15); // Bob
        BigInteger XC = BigInteger.valueOf(13); // Charlie

        System.out.println("Alice Priv = " + XA);
        System.out.println("Bob Priv = " + XB);
        System.out.println("Charlie Priv = " + XC);

        BigInteger q = BigInteger.valueOf(23);
        BigInteger g = BigInteger.valueOf(5);

        // public keys
        BigInteger YA = g.modPow(XA, q);
        BigInteger YB = g.modPow(XB, q);
        BigInteger YC = g.modPow(XC, q);

        System.out.println("\nPublic Keys:");
        System.out.println("Alice = " + YA);
        System.out.println("Bob = " + YB);
        System.out.println("Charlie = " + YC);

        // exchanging and compute intermediate values
        BigInteger AB = YB.modPow(XA, q); // Alice with Bob
        BigInteger BC = YC.modPow(XB, q); // Bob with Charlie
        BigInteger CA = YA.modPow(XC, q); // Charlie with Alice

        System.out.println("\nIntermediate Keys:");
        System.out.println("Alice-Bob = " + AB);
        System.out.println("Bob-Charlie = " + BC);
        System.out.println("Charlie-Alice = " + CA);

        // Final shared key 
        BigInteger K_A = BC.modPow(XA, q);
        BigInteger K_B = CA.modPow(XB, q);
        BigInteger K_C = AB.modPow(XC, q);

        System.out.println("\nFinal Shared Key:");
        System.out.println("Alice = " + K_A);
        System.out.println("Bob = " + K_B);
        System.out.println("Charlie = " + K_C);
    }
}

// DSA 
public class DSA {
    public static void main(String[] args) {
        // p, q, g
        BigInteger p = BigInteger.valueOf(23);
        BigInteger q = BigInteger.valueOf(11);
        BigInteger g = BigInteger.valueOf(2);

        // Private key
        BigInteger a = BigInteger.valueOf(6);

        // Public key
        BigInteger A = g.modPow(a, p);

        System.out.println("Private key x = " + a);
        System.out.println("Public key y = " + A);

        // Message hash
        BigInteger H = BigInteger.valueOf(9);

        // Random k
        BigInteger k = BigInteger.valueOf(3);

        // Signature generation
        BigInteger r = g.modPow(k, p).mod(q);
        BigInteger kInv = k.modInverse(q);
        BigInteger s = (kInv.multiply(H.add(a.multiply(r)))).mod(q);

        System.out.println("\nSignature:");
        System.out.println("r = " + r);
        System.out.println("s = " + s);

        // Verification
        BigInteger w = s.modInverse(q);
        BigInteger u1 = (H.multiply(w)).mod(q);
        BigInteger u2 = (r.multiply(w)).mod(q);

        BigInteger v = (g.modPow(u1, p).multiply(A.modPow(u2, p))).mod(p).mod(q);

        System.out.println("\nVerification value v = " + v);

        if (v.equals(r))
            System.out.println("Signature Verified");
        else
            System.out.println("Signature Invalid");
    }
}





